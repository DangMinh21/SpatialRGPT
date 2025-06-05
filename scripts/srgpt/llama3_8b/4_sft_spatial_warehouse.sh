#!/bin/bash

# --- GPU Configuration (Adapt for your RunPod setup) ---
# For a single A40 GPU on a single node:
NNODES=1
NPROC_PER_NODE=1 # Number of GPUs you want to use on this node
MASTER_PORT=25001 # Or any other free port

# --- Model and Data Paths (MUST BE SET CORRECTLY on RunPod) ---
# Path to the downloaded Stage 2 pre-trained model checkpoint
# This should be the output of a script like '2_pretrain.sh'
# Example: "./checkpoints/vila-siglip-llama3-8b-vila-v1.5-srgpt-pretrain"
# Or the Hugging Face ID if it represents this stage accurately.
# For your previous inference tests, you used "a8cheng/SpatialRGPT-VILA1.5-8B".
# Ensure this is the correct base for SFT.
PRETRAINED_MODEL_PATH="a8cheng/SpatialRGPT-VILA1.5-8B" # Or your local path to the stage 2 model

# Name of your AI City dataset mixture defined in datasets_mixture.py
# This name should point to your processed train_aicity_srgpt.jsonl and related image/depth folders
AICITY_DATA_MIXTURE_NAME="PSIW_sft_train" # Using the name from your datasets_mixture.py

# Output directory for your AI City fine-tuned model
OUTPUT_DIR="./checkpoints/spatialrgpt-aicity-qlora-A40-run1"

# --- Training Hyperparameters (Adjust for A40 and dataset size) ---
# Batch size per GPU. A40 has ~40-48GB. 8B model with ZeRO-3.
# Start very low and increase if memory allows.
PER_DEVICE_TRAIN_BATCH_SIZE=2 # Start with 1 or 2 for an 8B model on a single A40
GRADIENT_ACCUMULATION_STEPS=8 # Adjust to get a reasonable effective batch size
# Effective Batch Size = PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS
# E.g., 2 * 1 * 8 = 16. (The original 3_sft.sh had effective BS of 256 with 8 GPUs)

NUM_TRAIN_EPOCHS=1 # Fine-tuning on a specific domain might need a few epochs
LEARNING_RATE=2e-4   # Common SFT learning rate

# Vision Tower (Should match the pre-trained model)
VISION_TOWER="google/siglip-so400m-patch14-384"

# --- QLoRA Specific Parameters ---
BITS=4                     # Enable 4-bit quantization for QLoRA
LORA_ENABLE=True           # Enable LoRA
LORA_R=64                  # LoRA rank (common values: 8, 16, 32, 64)
LORA_ALPHA=16              # LoRA alpha (often 2*lora_r or lora_r)
LORA_DROPOUT=0.05          # LoRA dropout
DOUBLE_QUANT=True          # Use double quantization (QLoRA specific)
QUANT_TYPE="nf4"           # Quantization type: "nf4" (NormalFloat4) or "fp4"

# --- Tunable Components with QLoRA ---
# For QLoRA primarily on LLM:
TUNE_LLM_LORA=True          # Apply LoRA to the LLM
TUNE_VISION_TOWER=False     # Typically freeze vision tower, or full tune if memory allows and needed
TUNE_MM_PROJECTOR=True      # Often beneficial to tune the projector
TUNE_REGION_EXTRACTOR=True  # Also beneficial for spatial tasks

# --- Environment Variables ---
export WANDB_PROJECT="AI_City_Challenge_SpatialRGPT_FineTune" # Customize for your WandB

# --- Setup for torchrun (single node) ---
export MASTER_ADDR="127.0.0.1" 
# For single node, torchrun usually handles RANK and WORLD_SIZE based on nproc_per_node.
# Explicitly setting for clarity if needed by deeper scripts, but often not required for torchrun single-node.
# export RANK="0" 
# export WORLD_SIZE=$NPROC_PER_NODE

echo "Starting QLoRA Fine-tuning for AI City Challenge on A40..."
echo "  Number of Nodes: $NNODES"
echo "  Number of GPUs per Node: $NPROC_PER_NODE"
echo "  Pre-trained Model Path: $PRETRAINED_MODEL_PATH"
echo "  Data Mixture: $AICITY_DATA_MIXTURE_NAME"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Per Device Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Effective Batch Size: $(($PER_DEVICE_TRAIN_BATCH_SIZE * $NPROC_PER_NODE * $GRADIENT_ACCUMULATION_STEPS))"
echo "  Number of Epochs: $NUM_TRAIN_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  QLoRA: Enabled (Bits: $BITS, LoRA R: $LORA_R, LoRA Alpha: $LORA_ALPHA)"

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# The main training command using torchrun
# Ensure you are in the root of the SpatialRGPT cloned directory when running this script
torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$PRETRAINED_MODEL_PATH" \
    --version llama_3 \
    --data_mixture "$AICITY_DATA_MIXTURE_NAME" \
    --vision_tower "$VISION_TOWER" \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --enable_region True \
    --enable_depth True \
    --region_extractor regiongpt \
    --bits $BITS \
    --lora_enable $LORA_ENABLE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_llm $TUNE_LLM_LORA \
    --double_quant $DOUBLE_QUANT \
    --quant_type "$QUANT_TYPE" \
    --tune_vision_tower $TUNE_VISION_TOWER \
    --tune_mm_projector $TUNE_MM_PROJECTOR \
    --tune_language_model False \
    --tune_region_extractor $TUNE_REGION_EXTRACTOR \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb

echo "Fine-tuning script for AI City Challenge finished."