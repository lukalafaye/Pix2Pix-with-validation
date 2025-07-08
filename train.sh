#!/bin/bash

set -a
source .env
set +a

# Check if WANDB_API_KEY is set (either in environment or in .env file)
if [ -z "${WANDB_API_KEY}" ]; then
    echo "Warning: WANDB_API_KEY is not set. WandB logging may not work properly."
    echo "Please set WANDB_API_KEY in your .env file or environment."
    
    # Uncomment this line if you want to run without wandb
    # export WANDB_MODE="offline"
fi

export MODEL_NAME="timbrooks/instruct-pix2pix"
export DATASET_ID="lukalafaye/NoC_with_dots"

# Print environment info
echo "Running with configuration:"
echo "- Model: $MODEL_NAME"
echo "- Dataset: $DATASET_ID"
echo "- WandB enabled: ${WANDB_MODE:-yes}"
echo "- Output directory: flexgen_diffusion"
echo

ulimit -n 65000 && accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip --use_dataset_validation \
    --disable_safety_checker \
    --train_batch_size=64 --gradient_accumulation_steps=2 --gradient_checkpointing \
    --checkpointing_steps=500 \
    --learning_rate=1e-05 --max_grad_norm=1 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --report_to=wandb \
    --cache_dir="/data1/code/luka/instruct_pix2pix/cache" \
    --output_dir="flexgen_diffusion" \
    --num_train_epochs=1 \
    --validation_batches=10 \
    --num_validation_images=1000 \
    --lr_scheduler="cosine" \
    --resume_from_checkpoint="latest" \
    --lr_warmup_steps=500 \
    --push_to_hub
    
    # The checkpoint resume fix is now implemented in train_instruct_pix2pix.py
    # It will handle the case where checkpoint global_step > max_train_steps
    
    #    --resume_from_checkpoint="latest" \

    # \
    # --val_image_url="validation/input.png" \
    # --validation_prompt="validation/prompt.txt" \
    #--max_train_steps=15000 \
    #--max_train_samples=10 \
    #     --resume_from_checkpoint="latest" \

# train is 68700 examples, val is 8900 examples


# steps = num backward pass
# effective batch = gradient acc * train batch = 4
# 1 step = saw one effective batch
# 1 epoch = num samples (60k) / effective batch (4) = 15k steps
# validation every 1500 batches = 1500 steps?