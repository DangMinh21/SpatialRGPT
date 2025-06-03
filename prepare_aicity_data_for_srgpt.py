import json
import os
import re 

DEFAULT_IMAGE_TOKEN = "<image>"

def preprocess_aicity_conversations_for_script(conversations_list):
    """
    Modifies conversations:
    1. Prepends '<image>\n' to the first human turn's value.
    2. Replaces '<mask>' with '<mask> <depth>' in all human turns' values.
    """
    processed_conversations = []
    is_first_human_turn_overall = True  # To ensure <image>\n is only added once at the very beginning
    for i, turn in enumerate(conversations_list):
        new_turn = turn.copy()
        if new_turn.get("from") == "human":
            current_value = new_turn["value"]
            # Add <image>\n to the start of the first human utterance in the conversation
            if is_first_human_turn_overall:
                if not current_value.strip().startswith(DEFAULT_IMAGE_TOKEN):  # DEFAULT_IMAGE_TOKEN is "<image>"
                    current_value = DEFAULT_IMAGE_TOKEN + "\n" + current_value
                is_first_human_turn_overall = False

            # Replace <mask> with <mask> <depth>
            current_value = re.sub(r"<mask>", "<mask> <depth>", current_value)
            new_turn["value"] = current_value
        processed_conversations.append(new_turn)
    return processed_conversations


def convert_aicity_to_spatialrgpt_format(aicity_json_path, output_json_path):

    # load anotation file (json)
    print(f"Loading AI City data from: {aicity_json_path}")
    with open(aicity_json_path, "r") as f:
        aicity_data = json.load(f)

    print(f"Found {len(aicity_data)} samples. Converting...")
    skipped_samples = 0

    # open output file to write
    with open(output_json_path, "w") as outfile:
        # loop each sample
        for sample_idx, sample in enumerate(aicity_data):
            try:
                # take image
                image_filename_ext = sample.get("image")
                if not image_filename_ext or not isinstance(image_filename_ext, str):
                    print(
                        f"Warning: Skipping sample ID {sample.get('id')} (index {sample_idx}) due to missing/invalid 'image' field."
                    )
                    skipped_samples += 1
                    continue

                # take basename of image path: 000001.png -> 000001
                filename_base, _ = os.path.splitext(os.path.basename(image_filename_ext))

                # take conversation
                if (
                    not sample.get("conversations")
                    or not isinstance(sample["conversations"], list)
                    or not sample["conversations"]
                ):
                    print(
                        f"Warning: Skipping sample ID {sample.get('id')} (index {sample_idx}) due to missing/invalid 'conversations'."
                    )
                    skipped_samples += 1
                    continue

                modified_conversations = preprocess_aicity_conversations_for_script(sample["conversations"])

                # take rle msk
                rle_data = sample.get("rle", [])  # Default to empty list if missing
                if not isinstance(rle_data, list):
                    print(
                        f"Warning: RLE data for sample ID {sample.get('id')} (index {sample_idx}) is not a list. Using empty list."
                    )
                    rle_data = []

                # merge and write to output file
                formatted_sample = {
                    "id": sample["id"],
                    "image_base_filename": filename_base,  # For AICityLazySpatialDataset
                    "conversations": modified_conversations,
                    "rle": rle_data,
                    "category": sample.get("category", "unknown"),  # Keep for reference
                }
                outfile.write(json.dumps(formatted_sample) + "\n")
            except Exception as e:
                print(f"Error processing sample ID {sample.get('id')} (index {sample_idx}): {e}")
                skipped_samples += 1

    print(f"Conversion complete for {aicity_json_path}. Output saved to: {output_json_path}")
    if skipped_samples > 0:
        print(f"Skipped {skipped_samples} samples due to issues.")


if __name__ == "__main__":
    DEFAULT_IMAGE_TOKEN = "<image>"

    base_raw_data_dir = "PhysicalAI-Spatial-Intelligence-Warehouse" 
    # original_train_json = os.path.join(base_raw_data_dir, "train.json")
    original_train_json = os.path.join(base_raw_data_dir, "train_sample.json")
    original_val_json = os.path.join(base_raw_data_dir, "val.json")

    # Paths for processed output data
    processed_data_dir = "PhysicalAI-Spatial-Intelligence-Warehouse/formatted_dataset"  # Script will create this
    os.makedirs(processed_data_dir, exist_ok=True)

    processed_train_jsonl = os.path.join(processed_data_dir, "train_aicity_srgpt.jsonl")
    processed_val_jsonl = os.path.join(processed_data_dir, "val_aicity_srgpt.jsonl")

    print("Starting AI City Dataset Conversion for SpatialRGPT fine-tuning...")

    if os.path.exists(original_train_json):
        convert_aicity_to_spatialrgpt_format(original_train_json, processed_train_jsonl)
    else:
        print(f"ERROR: AI City train.json not found at {original_train_json}. Please check the path.")

    if os.path.exists(original_val_json):
        convert_aicity_to_spatialrgpt_format(original_val_json, processed_val_jsonl)
    else:
        print(f"ERROR: AI City val.json not found at {original_val_json}. Please check the path.")

    print("Dataset conversion script finished.")
