# InstructPix2Pix Training

This repository contains code for training and running the InstructPix2Pix model for instruction-based image editing, based on the original paper "InstructPix2Pix: Learning to Follow Image Editing Instructions" (https://arxiv.org/abs/2211.09800).

## Repository Structure

The codebase has been refactored into a modular structure to improve maintainability, readability, and code reuse:

```
instruct_pix2pix/
├── train_instruct_pix2pix.py  # Main training script
├── inference.py               # Standalone inference script
├── train.sh                   # Training shell script with example parameters
├── utils/                     # Modular utility functions
│   ├── __init__.py            # Exports all utility functions
│   ├── data_utils.py          # Dataset preprocessing and dataloaders
│   ├── image_processing.py    # Image conversion and manipulation
│   ├── inference.py           # Model inference functions
│   ├── logging_utils.py       # Logging and visualization utilities
│   ├── model_utils.py         # Model setup, configuration, and optimization
│   ├── training_utils.py      # Training helpers, checkpointing, and resuming
│   └── validation.py          # Validation metrics and testing
```

## Key Features

- **Memory-optimized training**: Uses gradient checkpointing, mixed precision, and efficient caching
- **Comprehensive validation**: Supports both single-image and full-dataset validation
- **EMA model support**: Exponential moving average model for better stability
- **Resumable training**: Robust checkpoint saving and loading
- **Extensive logging**: Integration with WandB for experiment tracking
- **Flexible dataset handling**: Works with HuggingFace datasets or local image folders

## Running the Code

### Training

```bash
python train_instruct_pix2pix.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="fusing/instructpix2pix-1000-samples" \
  --resolution=256 \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=5e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_ema \
  --validation_prompt="make the flower red" \
  --val_image_url="https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg" \
  --validation_batches=10 \
  --checkpointing_steps=5000 \
  --output_dir="instruct-pix2pix-model"
```

Or use the provided `train.sh` script with pre-configured parameters.

### Inference

```bash
python inference.py \
  --model_path="instruct-pix2pix-model" \
  --prompt="make the flower red" \
  --image_path="path/to/input/image.jpg" \
  --output_path="path/to/output/image.jpg"
```

## Hardware Requirements

- Training: NVIDIA GPU with at least 16GB VRAM recommended
- Inference: NVIDIA GPU with at least 8GB VRAM

## Dependencies

Main dependencies include:
- PyTorch
- Diffusers
- Transformers
- Accelerate
- Datasets
- WandB (optional for logging)

## References

- Original paper: [InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
