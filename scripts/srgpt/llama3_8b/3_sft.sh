#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

# OUTPUT of stage 2 script
# STAGE2_PATH="./checkpoints/vila-siglip-llama3-8b-vila-v1.5-srgpt-pretrain"
STAGE2_PATH="a8cheng/SpatialRGPT-VILA1.5-8B" 


n_node=$SLURM_JOB_NUM_NODES
bs=2
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $STAGE2_PATH \
    --version llama_3 \
    --data_mixture PSIW_sft_train \
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
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/vila-siglip-llama3-8b-vila-v1.5-srgpt-sft \
    --num_train_epochs 1 \
    --max_steps 10
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb