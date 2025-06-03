export MASTER_ADDR="127.0.0.1"
export WANDB_PROJECT="AI_City_SpatialRGPT" # Optional for wandb

# Assuming you are in the cloned SpatialRGPT directory
# and your prepared AI City dataset JSON is at /path/to/your_data/train_aicity.jsonl
# and images are in /path/to/your_data/ (with train/images/, train/depths/ subdirs)

torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=0 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/vila-siglip-llama3-8b-vila-v1.5-srgpt-pretrain \ # Path to YOUR downloaded base model
    --version llama_3 \
    --data_path /path/to/your_data/train_aicity.jsonl \ # Path to your AI City formatted JSON
    --image_folder /path/to/your_data/ \             # Base for RGB images
    --depth_image_folder /path/to/your_data/ \     # Base for Depth images (custom arg you'll add to train.py)
    --use_rle_masks True \                           # Custom arg for RLE loading (custom arg you'll add to train.py)
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
    --image_aspect_ratio pad \ # Or 'resize', ensure consistency with pre-training & model config
    --bf16 True \
    --output_dir ./checkpoints/spatialrgpt-aicity-sft-run1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \ # Reduced for potential memory constraints
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \ # Increased to maintain effective batch size
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \ # Save more frequently initially
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \ # Adjust based on RunPod vCPUs
    --lazy_preprocess True \
    --vflan_no_system_prompt True \ # From original script, keep if relevant for VILA/LLaMA3 base
    --report_to wandb # Optional