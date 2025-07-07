#!/bin/bash

set -a
source .env
set +a

export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_ID="lukalafaye/NoC_with_dots"

ulimit -n 65000 && accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip --use_dataset_validation \
    --disable_safety_checker \
    --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --checkpointing_steps=10000 \
    --learning_rate=1e-05 --max_grad_norm=1 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --report_to=wandb \
    --cache_dir="/data1/code/luka/instruct_pix2pix/cache" \
    --output_dir="flexgen_diffusion" \
    --num_train_epochs=2 \
    --validation_batches=150 \
    --num_validation_images=250 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=500 \
    --resume_from_checkpoint="latest" \
    --push_to_hub
    
    
    # \
    # --val_image_url="validation/input.png" \
    # --validation_prompt="validation/prompt.txt" \
    #--max_train_steps=15000 \
    #--max_train_samples=10 \
    #     --resume_from_checkpoint="latest" \



# steps = num backward pass
# effective batch = gradient acc * train batch = 4
# 1 step = saw one effective batch
# 1 epoch = num samples (60k) / effective batch (4) = 15k steps
# validation every 1500 batches = 1500 steps?