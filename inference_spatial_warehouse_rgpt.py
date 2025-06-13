import torch
import os
import sys
import json
import re
from PIL import Image
from tqdm import tqdm
from typing import Optional, Dict, Any

# --- Add Project Root to Path ---
# This ensures Python can find the 'llava' module
SPATIAL_RGPT_REPO_ROOT = "." 
sys.path.insert(0, os.path.abspath(SPATIAL_RGPT_REPO_ROOT))

# --- Necessary Imports ---
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from llava.train.args import DataArguments
    from llava.model.builder import load_pretrained_model
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import get_model_name_from_path, process_image, process_depth, process_masks, tokenizer_image_token
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.train.transformer_normalize_monkey_patch import patch_normalize_preprocess
    from word2number import w2n
except ImportError as e:
    print(f"ERROR: Failed to import components. Check paths and environment: {e}")
    sys.exit(1)

# Apply the normalization patch for consistency
patch_normalize_preprocess()

# --- Configuration ---
# 1. Path to your FINE-TUNED and MERGED model on the Hugging Face Hub
FINETUNED_VLM_ID = "DangMinh21/SpatialRGPT-VILA1.5-8B-SFT-SpatialWarehouse-merged"
MODEL_NAME = "SpatialRGPT-VILA1.5-8B-SFT-SpatialWarehouse-merged"

# 2. Path to your QUESTION CATEGORY CLASSIFIER on the Hugging Face Hub
CATEGORY_CLASSIFIER_ID = "DangMinh21/category_classifier_model"

# 3. Paths to the AI City Challenge validation data
# The script will iterate through your processed JSONL, but needs the raw ground truth for scoring
# and the image/depth folders for inference.
# PROCESSED_VAL_JSONL_PATH = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/formatted_dataset/val_aicity_srgpt.jsonl"
# RAW_VAL_JSON_PATH = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/val.json" # The original val.json with ground truth normalized_answer
# RGB_IMAGE_BASE_DIR = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/val/images/"
# DEPTH_IMAGE_BASE_DIR = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/val/depths/"
# OUTPUT_DIR = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/outputs/"

PROCESSED_VAL_JSONL_PATH = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/formatted_dataset/test_aicity_srgpt.jsonl"
RAW_VAL_JSON_PATH = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/test.json" # The original val.json with ground truth normalized_answer
RGB_IMAGE_BASE_DIR = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/test/images/"
DEPTH_IMAGE_BASE_DIR = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/test/depths/"
OUTPUT_DIR = "./datasets/PhysicalAI-Spatial-Intelligence-Warehouse/outputs/"

# --- Output Parsers (from previous step) ---
str_to_int = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

def parse_distance_output(text: str) -> Optional[float]:
    match = re.search(r"(\d+\.\d+|\d+)\s*(?:meters?|m)\b", text, re.IGNORECASE)
    if match:
        try: return float(match.group(1))
        except (ValueError, IndexError): pass
    match_any = re.search(r"(\d+\.\d+|\d+)", text)
    if match_any:
        try: return float(match_any.group(1))
        except (ValueError, IndexError): pass
    return None

def parse_count_output(text: str) -> Optional[int]:
    pattern = r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:pallets|pallet|buffer|buffersb)\b'
    matches = re.findall(pattern, text.lower())
    if not matches: return None
    for count_str in matches:
        if count_str in str_to_int: return str_to_int[count_str]
        try: return int(count_str)
        except ValueError: continue
    return None

def parse_mcq_output(text: str) -> Optional[int]:
    match = re.search(r'\[Region (\d+)\]\s+is\s+(?:nearest|the nearest|the leftmost|the shortest|the closest|the rightmost)', text)
    if match: return int(match.group(1))
    match_direct = re.search(r"(?i)\b(?:is|answer is|choice is)\s*(\d+)\b", text)
    if match_direct: return int(match_direct.group(2))
    return None

def parse_left_right_output(text: str) -> Optional[str]:
    text_lower = text.lower()
    is_negated = "not" in text_lower or "incorrect" in text_lower
    is_left, is_right = "left" in text_lower, "right" in text_lower
    if is_left and not is_right: return "right" if is_negated else "left"
    if is_right and not is_left: return "left" if is_negated else "right"
    return None

def get_parsed_answer(model_freeform_text: str, question_category: str):
    parser_map = {"distance": parse_distance_output, "count": parse_count_output, "mcq": parse_mcq_output, "left_right": parse_left_right_output}
    parser_func = parser_map.get(question_category)
    if parser_func: return parser_func(model_freeform_text)
    return None

# --- Scoring Function (Official Logic) ---

def check_success(parsed_answer: Any, ground_truth: Any, category: str) -> bool:
    if parsed_answer is None: return False
    if category in ["distance", "count"]:
        try:
            gt_val, pred_val = float(ground_truth), float(parsed_answer)
        except (ValueError, TypeError): return False
        if gt_val == 0: return abs(pred_val) <= 0.10
        return abs(pred_val - gt_val) / abs(gt_val) <= 0.10
    else: # mcq, left_right, yes_no
        return str(parsed_answer).lower() == str(ground_truth).lower()

# --- Inference Function ---
def get_prediction_for_sample(sample_dict: Dict, tokenizer, model, data_args: DataArguments):
    image_base_filename = sample_dict['image_base_filename']
    rgb_path = os.path.join(RGB_IMAGE_BASE_DIR, image_base_filename + ".png")
    depth_path = os.path.join(DEPTH_IMAGE_BASE_DIR, f"{image_base_filename}_depth.png")
    
    try:
        rgb_tensor, image_info = process_image(rgb_path, data_args, image_folder=None, return_info=True)
        depth_tensor = process_depth(depth_path, data_args, depth_folder=None)
        masks_tensor = process_masks([sample_dict], data_args, image_info)
    except Exception as e:
        print(f"Error processing visuals for sample {sample_dict['id']}: {e}")
        return "Error processing inputs."

    conv = conv_templates[data_args.conv_mode].copy()
    question_text = sample_dict['conversations'][0]['value']
    conv.append_message(conv.roles[0], question_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(model.device)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            images=rgb_tensor.unsqueeze(0).to(dtype=torch.float16, device=model.device),
            depths=depth_tensor.unsqueeze(0).to(dtype=torch.float16, device=model.device),
            masks=[masks_tensor.to(dtype=torch.float16, device=model.device)],
            do_sample=False,
            # temperature=0,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True
        )
    
    decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if stop_str and decoded_output.endswith(stop_str):
        decoded_output = decoded_output[:-len(stop_str)]
    return decoded_output.strip()

# --- Main Evaluation Logic ---

def evaluate():
    # 1. Load Models and Tokenizers
    print(f"====================== Load Model ===========================")
    print(f"Loading fine-tuned VLM from: {FINETUNED_VLM_ID}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        FINETUNED_VLM_ID, model_name=MODEL_NAME, model_base=None, trust_remote_code=True
    )
    
    model.to(device='cuda', dtype=torch.float16).eval()
    print("Fine-tuned VLM loaded successfully.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer `pad_token` was None. Set to `eos_token`: '{tokenizer.eos_token}'")
    
    # 2. Update the model's `generation_config`. This is the object checked by model.generate().
    #    The `model` object from `load_pretrained_model` is the top-level LlavaLlamaModel.
    #    This top-level model has its own `generation_config`.
    if hasattr(model, 'generation_config'):
        print(f"Updating model's top-level generation_config.pad_token_id...")
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        # It's also good practice to ensure the eos token is consistent
        if model.generation_config.eos_token_id is None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
    else:
        print("Warning: Top-level model does not have a `generation_config` attribute. Creating one.")
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_model_config(model.config)
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        
    print("Pad token ID configuration updated for model.generate().")

    print(f"Loading category classifier from Hub: {CATEGORY_CLASSIFIER_ID}")
    category_classifier = pipeline("text-classification", model=CATEGORY_CLASSIFIER_ID, device=0)
    print("Category classifier pipeline loaded successfully.")

    # 2. Setup DataArguments
    # data_args = DataArguments(image_processor=image_processor, image_aspect_ratio='pad', is_multimodal=True, conv_mode='llava_v1')
    data_args = DataArguments()
    data_args.image_processor = image_processor
    data_args.conv_mode='llama_3'
    data_args.is_multimodel=True
    
    # 3. Load Datasets
    print(f"====================== Load Dataset ===========================")
    
    print(f"Loading processed validation data from: {PROCESSED_VAL_JSONL_PATH}...")
    with open(PROCESSED_VAL_JSONL_PATH, 'r') as f:
        val_data_processed = [json.loads(line) for line in f]
    
    # print(f"Loading raw validation data from: {RAW_VAL_JSON_PATH}...")
    # with open(RAW_VAL_JSON_PATH, 'r') as f:
    #     val_data_raw = json.load(f)
    
    # gt_lookup = {item['id']: {"normalized_answer": item['normalized_answer'], "category": item['category']} for item in val_data_raw}
    # print(f"Loaded {len(val_data_processed)} validation samples to evaluate.")

    # 4. Initialize Evaluation Counters & Results Storage
    # category_scores = {}
    evaluation_results = []

    # 5. Main Evaluation Loop
    print(f"====================== Process Samples ===========================")
    
    for sample in tqdm(val_data_processed, desc="Evaluating Validation Set"):
        pred_id = sample.get("id")
        # gt_item = gt_lookup.get(pred_id)
        # if not gt_item: 
        #     continue

        question_text_for_classifier = sample['conversations'][0]['value'].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        
        # Step A: Classify category
        classifier_output = category_classifier(question_text_for_classifier, top_k=1)
        predicted_category = classifier_output[0]['label'] if classifier_output else "distance" # most frequency

        # Step B: Generate freeform answer
        model_prediction_text = get_prediction_for_sample(sample, tokenizer, model, data_args)
        
        # Step C: Parse the answer
        parsed_answer = get_parsed_answer(model_prediction_text, predicted_category)

        # Step D: Compare and Score
        # is_correct = check_success(parsed_answer, gt_item['normalized_answer'], gt_item['category'])
        
        # Store results
        # if gt_item['category'] not in category_scores:
        #     category_scores[gt_item['category']] = {'correct': 0, 'total': 0}
        
        # if is_correct: category_scores[gt_item['category']]['correct'] += 1
        # category_scores[gt_item['category']]['total'] += 1

        evaluation_results.append({
            "id": pred_id,
            # "category": gt_item['category'],
            "question": sample['conversations'][0]['value'],
            # "freeform_answer": sample['conversations'][1]['value'],
            "model_answer": model_prediction_text,
            "parsed_answer": parsed_answer,
            "category_pred": predicted_category,
            # "ground_truth": gt_item['normalized_answer'],
            # "is_correct": is_correct
        })

    # 6. Calculate and Display Final Metrics
    # print("\n--- AI City Challenge :: Fine-tuned Model Evaluation Summary ---")
    # overall_correct, overall_total = 0, 0
    # for category, scores in sorted(category_scores.items()):
    #     total, correct = scores['total'], scores['correct']
    #     accuracy = (correct / total) * 100 if total > 0 else 0
    #     print(f"Category: {category:<25} | Success Rate: {accuracy:>6.2f}% ({correct}/{total})")
    #     overall_correct += correct
    #     overall_total += total
    
    # if overall_total > 0:
    #     overall_accuracy = (overall_correct / overall_total) * 100
    #     print("---------------------------------------------------------------")
    #     print(f"Overall Average Success Rate: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")
    #     print("(Note: Final official score will be a *weighted* average)")
    # else:
    #     print("No samples were scored.")

    # 7. Save detailed results for error analysis
    print(f"Length of input file: {len(val_data_processed)}")
    print(f"Length of output file: {len(evaluation_results)}")
    output_results_file = OUTPUT_DIR + "evaluation_test.json"
    with open(output_results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {output_results_file}")

if __name__ == "__main__":
    evaluate()

# First run
# --- AI City Challenge :: Fine-tuned Model Evaluation Summary ---
# Category: count                     | Success Rate:  15.40% (77/500)
# Category: distance                  | Success Rate:  83.33% (405/486)
# Category: left_right                | Success Rate:  99.80% (499/500)
# Category: mcq                       | Success Rate:  40.57% (185/456)
# ---------------------------------------------------------------
# Overall Average Success Rate: 60.04% (1166/1942)