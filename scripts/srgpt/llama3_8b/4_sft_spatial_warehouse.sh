#!/bin/bash

NNODES=1
NPROC_PER_NODE=1
MASTER_PORT=25001 

PRETRAINED_MODEL_PATH="a8cheng/SpatialRGPT-VILA1.5-8B" 
echo $PRETRAINED_MODEL_PATH

AICITY_DATA_MIXTURE_NAME="PSIW_sft_train" 
echo $AICITY_DATA_MIXTURE_NAME

OUTPUT_DIR="./checkpoints/SpatialRGPT-VILA1.5-8B-SFT-SpatialWarehouse"
echo $OUTPUT_DIR

RESUME_CHECKPOINT_PATH="$OUTPUT_DIR/checkpoint-10989"
echo "Resuming training from checkpoint: $RESUME_CHECKPOINT_PATH"

PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1

NUM_TRAIN_EPOCHS=1 
LEARNING_RATE=2e-4 

VISION_TOWER="google/siglip-so400m-patch14-384"

# --- QLoRA Specific Parameters ---
BITS=4                     # Enable 4-bit quantization for QLoRA
LORA_ENABLE=True           # Enable LoRA
LORA_R=32                  # LoRA rank (common values: 8, 16, 32, 64)
LORA_ALPHA=32              # LoRA alpha (often 2*lora_r or lora_r)
LORA_DROPOUT=0.05          # LoRA dropout
DOUBLE_QUANT=True          # Use double quantization (QLoRA specific)
QUANT_TYPE="nf4"           # Quantization type: "nf4" (NormalFloat4) or "fp4"

TUNE_LLM_LORA=True          # Apply LoRA to the LLM
TUNE_VISION_TOWER=False     # Typically freeze vision tower, or full tune if memory allows and needed
TUNE_MM_PROJECTOR=True      # Often beneficial to tune the projector
TUNE_REGION_EXTRACTOR=True  # Also beneficial for spatial tasks
TUNE_LANGUAGE_MODEL=False

# --- Environment Variables ---
export WANDB_PROJECT="AI_City_Challenge_SpatialRGPT_FineTune" 

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

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
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
    --tune_language_model $TUNE_LANGUAGE_MODEL \
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
    --save_steps 999 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb \
    --resume_from_checkpoint $RESUME_CHECKPOINT_PATH

echo "Fine-tuning script for AI City Challenge finished."