#!/bin/bash

# --- RunPod Configuration (Single Node Multi-GPU) ---
# MASTER_ADDR will be set by torchrun for single node, or manually for multi-node
# MASTER_PORT can be any free port
NNODES=1 # Assuming a single RunPod instance
NPROC_PER_NODE=1 # Set this to the number of GPUs on your RunPod instance (e.g., 2, 4, 8)
MASTER_PORT=25001 # Or any other free port

# --- Model and Data Paths ---
# Path to the downloaded Stage 2 pre-trained model checkpoint
# Replace this with the actual path on your RunPod instance
PRETRAINED_CHECKPOINT_PATH="checkpoint/SpatialRGPT-VILA1.5-8B" 
# Or if using a HF model ID that represents this stage:
# PRETRAINED_CHECKPOINT_PATH="a8cheng/SpatialRGPT-VILA1.5-8B" # Ensure this is the STAGE 2 equivalent

# Name of your AI City dataset mixture defined in datasets_mixture.py
AICITY_DATA_MIXTURE_NAME="aicity_sft_train"

# Output directory for your fine-tuned model
OUTPUT_DIR="./checkpoints/spatialrgpt-aicity-sft-run1"

# --- Training Hyperparameters (Adjust as needed based on GPU memory and experimentation) ---
# Batch size per GPU. The original script used 16. This might be too high for an 8B model.
# Start low (e.g., 1, 2, or 4) and see if memory allows more.
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=4 # Adjust to maintain effective batch size
# Effective Batch Size = PER_DEVICE_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS
# Example: 4 * 8 * 4 = 128 (Original was 16 * 8 * 2 = 256)

NUM_TRAIN_EPOCHS=1 # Start with 1 epoch, can increase later
LEARNING_RATE=2e-5

# --- Ensure an environment variable for WandB project if you use it ---
export WANDB_PROJECT="AI_City_Challenge_SpatialRGPT"

echo "Starting Fine-tuning for AI City Challenge..."
echo "Number of Nodes: $NNODES"
echo "Number of GPUs per Node: $NPROC_PER_NODE"
echo "Pre-trained Model Path: $PRETRAINED_CHECKPOINT_PATH"
echo "Data Mixture: $AICITY_DATA_MIXTURE_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "Per Device Batch Size: $PER_DEVICE_BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective Batch Size: $(($PER_DEVICE_BATCH_SIZE * $NPROC_PER_NODE * $GRADIENT_ACCUMULATION_STEPS))"

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# The main training command (adapted from 3_sft.sh)
# Using localhost for MASTER_ADDR for single-node training
# torchrun will handle setting MASTER_ADDR for its processes if not set globally for single node
# For multi-node, MASTER_ADDR of the rank 0 node is needed.
# node_rank is 0 for single node.

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $PRETRAINED_CHECKPOINT_PATH \
    --version llama_3 \
    --data_mixture $AICITY_DATA_MIXTURE_NAME \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --enable_region True \
    --enable_depth True \
    --region_extractor regiongpt \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --tune_region_extractor True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \ # Changed to 'pad' for potentially better region consistency; 'resize' was in original 3_sft.sh. Test what works.
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 4 \ # Eval batch size can be different
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \ # Set to "steps" if you have a val set and want to eval during training
    --eval_steps 200 \ # Example: Evaluate every 200 steps if strategy is "steps"
    --save_strategy "steps" \
    --save_steps 20 \ # Save checkpoint every X steps
    --save_total_limit 2 \ # Keep only the last 2 checkpoints
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \ # Adjust based on your RunPod vCPUs (e.g., 2-4 per GPU)
    --lazy_preprocess True \
    --vflan_no_system_prompt True \ # Specific to VILA/LLaMA3 template from original script
    --report_to wandb # Or "none" if not using Weights & Biases

echo "Fine-tuning script finished."