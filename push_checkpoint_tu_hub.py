import torch
import os
import sys
from transformers import AutoTokenizer
from peft import PeftModel
from huggingface_hub import login, whoami, HfApi

# --- Ensure SPATIAL_RGPT_REPO_ROOT is set correctly for imports ---
# Assuming this script is in the /root/SpatialRGPT/ directory
SPATIAL_RGPT_REPO_ROOT = "." 
sys.path.insert(0, os.path.abspath(SPATIAL_RGPT_REPO_ROOT))

try:
    from llava.model.builder import load_pretrained_model
    print("Successfully imported `load_pretrained_model` from the project.")
except ImportError as e:
    print(f"ERROR: Failed to import from LLaVA. Check SPATIAL_RGPT_REPO_ROOT path: {e}")
    sys.exit(1)

# # --- Configuration: UPDATE THESE ---

# # 1. Path to the base model your fine-tuning started from.
BASE_MODEL_ID = "a8cheng/SpatialRGPT-VILA1.5-8B"
MODEL_NAME = "SpatialRGPT-VILA1.5-8B"

# # 2. Path to your fine-tuning output directory containing the LoRA adapters.
# #    This is the --output_dir from your fine-tuning script.
ADAPTER_CHECKPOINT_PATH = "./checkpoints/SpatialRGPT-VILA1.5-8B-SFT-SpatialWarehouse" # <-- Verify this path!

# # 3. Choose unique names for your new repositories on the Hub.
REPO_ID_ADAPTERS_ONLY = "SpatialRGPT-VILA1.5-8B-SFT-SpatialWarehouse-adapters" # <-- Customize if you like
REPO_ID_MERGED = "SpatialRGPT-VILA1.5-8B-SFT-SpatialWarehouse-merged"        # <-- Customize if you like

def main():
    print("========== Push model to Hub ==========")
    # --- Step 1: Log in to Hugging Face Hub ---
    print("Logging in to Hugging Face Hub. Please provide your write-access token.")
    login("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")
    
    try:
        hf_user = whoami()['name']
        print(f"\nSuccessfully logged in as: {hf_user}")
    except Exception as e:
        print(f"Login failed. Please check token. Error: {e}")
        return

    # Construct full repository IDs with your username
    hub_model_id_adapters = f"{hf_user}/{REPO_ID_ADAPTERS_ONLY}"
    hub_model_id_merged = f"{hf_user}/{REPO_ID_MERGED}"
    # print(f"Upload to: {hub_model_id_adapters}")
    # print(f"Upload to: {hub_model_id_merged}")
    
    # Create the repositories on the Hub first
    api = HfApi()
    api.create_repo(repo_id=hub_model_id_adapters, repo_type="model", exist_ok=True)
    api.create_repo(repo_id=hub_model_id_merged, repo_type="model", exist_ok=True)
    print(f"Created/verified Hub repositories:\n  - {hub_model_id_adapters}\n  - {hub_model_id_merged}")

    # --- Step 2: Load Base Model, Tokenizer, and Apply Adapters ---
    print("\nLoading base model, tokenizer, and processor...")
    # Use the project's own loader to correctly instantiate the custom architecture
    # Load in float16 for merging. No need for 4-bit quantization here.
    tokenizer, base_model, image_processor, _ = load_pretrained_model(
        model_path=BASE_MODEL_ID, 
        model_name=MODEL_NAME, # Inferred from path
        # model_base=None, 
        # load_8bit=False, load_4bit=False, 
        # torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        # trust_remote_code=True # Essential for custom model code
    )
    
    print(f"\nLoading PEFT adapters from: {ADAPTER_CHECKPOINT_PATH}")
    # This creates a PeftModel object which wraps the base model with the LoRA layers
    model_with_adapters = PeftModel.from_pretrained(
        base_model,
        ADAPTER_CHECKPOINT_PATH
    )
    print("Adapters loaded successfully.")

    # --- Step 3: Push Adapters-Only Repo ---
    # try:
    #     print("\n--- Strategy A: Pushing adapters-only to Hub ---")
    #     print(f"Uploading to: {hub_model_id_adapters}")
        
    #     # When you push a PeftModel object, it intelligently only uploads the adapter files.
    #     model_with_adapters.push_to_hub(
    #         hub_model_id_adapters,
    #         commit_message="Upload fine-tuned LoRA adapters for SpatialRGPT of Spatial Warehouse Dataset"
    #     )
    #     tokenizer.push_to_hub(hub_model_id_adapters, commit_message="Upload tokenizer")
    #     print(f"SUCCESS: Adapters and tokenizer uploaded to {hub_model_id_adapters}")
    # except Exception as e:
    #     print(f"\nERROR during adapter push: {e}")

    # --- Step 4: Merge Weights and Push Full Model ---
    try:
        print("\n--- Strategy B: Merging adapters and pushing full model ---")
        
        # Merge the LoRA weights into the base model. This returns a standard LlavaLlamaModel.
        merged_model = model_with_adapters.merge_and_unload()
        print("Model merging complete.")

        local_save_path = f"./{REPO_ID_MERGED}_local" # Create a temporary local folder
        print(f"Saving merged model to local directory: {local_save_path}")
        merged_model.save_pretrained(local_save_path)
        tokenizer.save_pretrained(local_save_path)

        # It's good practice to save the image_processor too
        if hasattr(base_model.get_vision_tower(), 'image_processor'):
            base_model.get_vision_tower().image_processor.save_pretrained(local_save_path)
        print("Model saved locally.")

        # 4c. Upload the contents of the local directory to the Hub
        print(f"Uploading folder contents to Hub repository: {hub_model_id_merged}...")
        api.upload_folder(
            folder_path=local_save_path,
            repo_id=hub_model_id_merged,
            repo_type="model",
            commit_message="Upload merged SFT model for AI City Challenge"
        )
        print(f"SUCCESS: Full merged model uploaded to {hub_model_id_merged}")
    except Exception as e:
        print(f"\nERROR during merged model push: {e}")
        
    print("\n\nAll tasks complete! Check your Hugging Face profile for the new repositories.")

# Ensure the checkpoint path exists before running
if not os.path.isdir(ADAPTER_CHECKPOINT_PATH):
    print(f"ERROR: Checkpoint directory not found at '{ADAPTER_CHECKPOINT_PATH}'")
    print("Please update the FINETUNED_ADAPTER_CHECKPOINT_PATH variable in this script.")
else:
    main()