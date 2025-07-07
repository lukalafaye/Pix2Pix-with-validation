#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import gc
import joblib
import logging
import math
import multiprocessing
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
import cv2
from datasets import DatasetDict, Image
import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from joblib import Parallel, delayed

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from huggingface_hub import HfFolder
token = HfFolder.get_token()

if is_wandb_available():
    import wandb
else:
    wandb = None

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]


def log_validation(
    unet,
    vae, 
    text_encoder,
    noise_scheduler,
    args,
    accelerator,
    generator,
    val_dataloader=None,
    tokenizer=None,
    epoch=None,
    global_step=None,
    weight_dtype=torch.float32
):
    # For real-time validation during training, use a small number of validation images
    # Default to 4-8 images for quick validation feedback during training
    if args.num_validation_images and args.num_validation_images > 0:
        max_val_images = args.num_validation_images
    else:
        # During training validation, use a small number for quick feedback
        max_val_images = 8  # Small number for real-time validation
    
    logger.info(f"Running validation on {max_val_images} samples...")
    
    # Memory debugging - before validation with aggressive cleanup
    if torch.cuda.is_available():
        # Force aggressive cleanup before validation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved_before = torch.cuda.memory_reserved() / 1024**3  # GB
        logger.info(f"MEMORY DEBUG - Before validation: Allocated={memory_before:.2f}GB, Reserved={memory_reserved_before:.2f}GB")
    
    # Set models to evaluation mode with explicit no_grad context
    unet.eval()
    vae.eval()
    text_encoder.eval()
    
    logger.info(f"Set models to eval mode: unet={unet.training}, vae={vae.training}, text_encoder={text_encoder.training}")

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    sum_matching_green = 0.0
    sum_avg_green = 0.0
    sum_matching_purple = 0.0
    sum_avg_purple = 0.0
    total_samples = 0

    # If val_dataloader is provided, use it; otherwise fall back to single image validation
    if val_dataloader is not None:

        with autocast_ctx, torch.no_grad():
            samples_processed = 0
            logger.info(f"Using validation dataloader with {len(val_dataloader)} batches")
            logger.info(f"Processing ENTIRE validation dataset for comprehensive validation...")
            
            if accelerator.is_main_process and wandb is not None:
                    columns = ["original_image", "predicted_visualization", "processed_visualization", "ground_truth", "edit_prompt"]
                    table = wandb.Table(columns=columns)

            for batch_idx, batch in enumerate(val_dataloader):
                logger.debug(f"Processing batch {batch_idx} in validation epoch {epoch}")

                # Stop at max images...
                if samples_processed >= max_val_images:
                    logger.info(f"Reached maximum validation samples: {max_val_images}. Stopping early.")
                    break
                    
                batch_size = batch["original_pixel_values"].shape[0]
                logger.info(f"Processing batch {batch_idx}/{len(val_dataloader)} with batch_size={batch_size}, image shapes={batch['original_pixel_values'].shape}")
                
                # Memory check before batch processing
                if torch.cuda.is_available():
                    memory_before_batch = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"MEMORY DEBUG - Before batch {batch_idx}: Allocated={memory_before_batch:.2f}GB")
                
                # Process batch - decode prompts first and immediately convert to CPU tensors
                batch_prompts = []
                for i in range(batch_size):
                    input_ids = batch["input_ids"][i]
                    prompt = tokenizer.decode(input_ids, skip_special_tokens=True).strip()
                    batch_prompts.append(prompt)
                
                # Move batch tensors to CPU immediately and delete originals
                original_pixel_values_cpu = batch["original_pixel_values"].cpu()
                edited_pixel_values_cpu = batch["edited_pixel_values"].cpu()
                input_ids_cpu = batch["input_ids"].cpu()
                
                # Delete the original batch to free GPU memory immediately
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                try:
                    logger.info(f"Attempting batch inference for {batch_size} images...")
                    
                    # Force memory cleanup before inference
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Use CPU tensors moved back to GPU only for inference
                    predicted_batch = run_diffusion_inference_batch(
                        unet, vae, text_encoder, noise_scheduler,
                        original_pixel_values_cpu.to(accelerator.device, dtype=weight_dtype),
                        batch_prompts,
                        tokenizer,
                        accelerator.device,
                        generator,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7
                    )
                    
                    # Images are already PIL Images on CPU from run_diffusion_inference_batch
                    predicted_batch_cpu = [img.copy() for img in predicted_batch]  # Ensure CPU copy

                    del predicted_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    logger.info(f"Batch inference successful for batch {batch_idx}")
                    
                    # Filter out completely black images using CPU tensors
                    filtered_indices = []
                    for idx, img in enumerate(predicted_batch_cpu):
                        extrema = img.getextrema()
                        if any(channel_ext[1] > 0 for channel_ext in extrema):
                            filtered_indices.append(idx)

                    # Process filtered results - predicted_batch_cpu contains PIL Images, others are tensors
                    filtered_predicted_images = [predicted_batch_cpu[idx] for idx in filtered_indices]  # Already PIL Images
                    filtered_gt_images = [tensor_to_pil(edited_pixel_values_cpu[idx]) for idx in filtered_indices]
                    filtered_prompts = [batch_prompts[idx] for idx in filtered_indices]
                    filtered_orig_images = [tensor_to_pil(original_pixel_values_cpu[idx]) for idx in filtered_indices]
                    
                    # Clear CPU tensors after processing
                    del original_pixel_values_cpu, edited_pixel_values_cpu, input_ids_cpu
                    del predicted_batch_cpu, batch_prompts
                    
                    # Process validation scores
                    filtered_validation_images = []

                    results = Parallel(n_jobs=4)(
                        delayed(calculate_validation_score)(pred_img, gt_img)
                        for pred_img, gt_img in zip(filtered_predicted_images, filtered_gt_images)
                    )

                    for scores in results:
                        filtered_validation_images.append(scores["visualization_img"])  # Already PIL image
                        sum_matching_green += scores["matching_score_green"]
                        sum_avg_green += scores["average_valid_score_green"]
                        sum_matching_purple += scores["matching_score_purple"]
                        sum_avg_purple += scores["average_valid_score_purple"]
                        total_samples += 1
                    
                    # Clear results immediately after processing
                    del results

                    logger.info(f"Batch {batch_idx} processed successfully with {len(filtered_validation_images)} valid images")

                    # log images to wandb
                    if batch_idx <= 1 and table is not None:
                        for i in range(len(filtered_orig_images)):
                            row_data = [
                                wandb.Image(filtered_orig_images[i]),
                                wandb.Image(filtered_predicted_images[i]),  # This is the visualization image
                                wandb.Image(filtered_validation_images[i]),
                                wandb.Image(filtered_gt_images[i]),
                                filtered_prompts[i]
                            ]
                            table.add_data(*row_data)
                
                        
                        # Clean up wandb table variables
                        del row_data
                        
                        # Force memory cleanup after WandB logging
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    elif batch_idx == 2:
                        # Create table name based on global step
                        images_table_name = f"validation_images_step_{global_step}" if global_step is not None else f"validation_images_epoch_{epoch}"
                        
                        # Use the same step_id as for metrics to ensure consistent step ordering
                        step_id = global_step if global_step is not None else 0
                
                        # Log the images table with explicit step using wandb.log directly
                        logger.info(f"Logging {len(table.data)} validation images to WandB table '{images_table_name}' at step {step_id}")
                        wandb.log({images_table_name: table}, step=step_id)
                        logger.info(f"Successfully logged {len(filtered_orig_images)} validation images to WandB table '{images_table_name}' at step {step_id}")
                        
                        del table, images_table_name

                    # Clean up batch variables to free memory
                    # Note: predicted_batch and results already deleted earlier
                    del filtered_indices
                    del filtered_predicted_images
                    del filtered_gt_images
                    del filtered_prompts
                    del filtered_orig_images
                    del filtered_validation_images
                    
                    # Force garbage collection and empty CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    samples_processed += batch_size
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    # Clean up even on error
                    if 'batch' in locals():
                        del batch
                    if 'predicted_batch' in locals():
                        del predicted_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                    
                # Log progress periodically
                if batch_idx % 10 == 0:
                    logger.info(f"Validation progress: {batch_idx}/{len(val_dataloader)} batches, {samples_processed} samples processed")
        
        logger.info(f"Validation loop finished for epoch {epoch}. Processed and plotted {samples_processed} samples to wandb.")
            
        # Memory debugging after validation loop
        if torch.cuda.is_available():
            memory_after_loop = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"MEMORY DEBUG - After validation loop: Allocated={memory_after_loop:.2f}GB")
        
        # Log metrics
        logger.info(f"Logging metrics for step: {global_step}")
        
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                if wandb is None:
                    logger.warning("WandB is not available, skipping validation table logging")
                    continue
                    
                try:
                    avg_metrics = {
                        "validation/green_mean_matching_score": sum_matching_green / total_samples if total_samples else float('nan'),
                        "validation/green_mean_average_score": sum_avg_green / total_samples if total_samples else float('nan'),
                        "validation/purple_mean_matching_score": sum_matching_purple / total_samples if total_samples else float('nan'),
                        "validation/purple_mean_average_score": sum_avg_purple / total_samples if total_samples else float('nan'),
                    }

                    wandb.log(avg_metrics, step=global_step)
                    
                    # Clean up metrics variables
                    del avg_metrics
                    
                    # Force memory cleanup after metrics logging
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Failed to log validation tables to WandB: {e}")
                    logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
    else:
        # Fallback to original single image validation
        logger.warning("No validation dataloader provided, using single image validation (metrics not available without ground truth)")
        original_image = download_image(args.val_image_url)
        edited_images = []
        
        with autocast_ctx:
            with torch.no_grad():
                # Convert original image to tensor format for batch processing
                original_tensor = torch.tensor(convert_to_np(original_image, args.resolution)).unsqueeze(0)
                original_tensor = 2 * (original_tensor / 255) - 1  # Normalize to [-1, 1]
                
                # Create batch of prompts
                prompts_batch = [args.validation_prompt] * min(args.num_validation_images or 4, 4)
                
                try:
                    # Use batch inference for efficiency
                    predicted_batch = run_diffusion_inference_batch(
                        unet, vae, text_encoder, noise_scheduler,
                        original_tensor.repeat(len(prompts_batch), 1, 1, 1).to(accelerator.device, dtype=weight_dtype),
                        prompts_batch,
                        tokenizer,
                        accelerator.device,
                        generator,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7
                    )
                    edited_images = predicted_batch[:args.num_validation_images or 4]
                    
                    # Clean up fallback validation variables
                    del predicted_batch, original_tensor, prompts_batch
                    
                    # Force memory cleanup after fallback inference
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to run batch inference for fallback validation: {e}")
                    # Create dummy images as fallback
                    for _ in range(args.num_validation_images or 4):
                        edited_images.append(original_image.copy())

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                if wandb is None:
                    logger.warning("WandB is not available, skipping image logging")
                    continue
                    
                try:                    # Use consistent table structure with main validation path
                    columns = ["original_image", "predicted_visualization", "edit_prompt"]
                    wandb_table = wandb.Table(columns=columns)
                    
                    for edited_image in edited_images:
                        # Ensure images are PIL Images
                        if not isinstance(original_image, PIL.Image.Image):
                            logger.warning(f"Original image is not PIL.Image, type: {type(original_image)}")
                            continue
                        if not isinstance(edited_image, PIL.Image.Image):
                            logger.warning(f"Edited image is not PIL.Image, type: {type(edited_image)}")
                            continue
                            
                        # Images only table (no metrics for fallback validation since no ground truth)
                        row_data = [
                            wandb.Image(original_image),
                            wandb.Image(edited_image),  # Use predicted image as visualization since no ground truth
                            args.validation_prompt
                        ]
                        wandb_table.add_data(*row_data)
                    
                    # Create table name for fallback validation (images only)
                    table_name = f"validation_images_epoch_{epoch}" if epoch is not None else "validation_images_final"
                    
                    # Use consistent step_id for fallback validation
                    step_id = global_step if global_step is not None else 0
                    
                    # Log with explicit step using wandb.log directly
                    wandb.log({table_name: wandb_table}, step=step_id)
                    logger.info(f"Successfully logged {len(edited_images)} fallback validation images to WandB table '{table_name}' at step {step_id}")
                    
                    # Clean up fallback WandB variables
                    del wandb_table, columns, table_name
                    
                    # Force memory cleanup after fallback WandB logging
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to log fallback validation images to WandB: {e}")
                    logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
    
    # Final memory cleanup at end of validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Final memory debug info
        memory_after_validation = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"MEMORY DEBUG - After complete validation: Allocated={memory_after_validation:.2f}GB")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument( # not used
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument( # not used because dataset already uses input_image
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument( # not used because dataset already uses edited_image
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument( # not used because dataset already uses edit_prompt
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=None,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1000,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://huggingface.co/papers/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8, # max num workers!
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb", # default wandb
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--use_dataset_validation",
        action="store_true",
        help="Use validation dataset for comprehensive validation instead of single image validation.",
    )
    parser.add_argument(
        "--disable_safety_checker",
        action="store_true",
        help="Disable NSFW safety checker during validation to avoid blocking potentially benign content.",
    )
    parser.add_argument(
        "--validation_batches",
        type=int,
        default=1000,
        help="Perform validation every N batches during training.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision
    
    if args.validation_prompt and args.validation_prompt.endswith(".txt"):
        with open(args.validation_prompt, "r") as f:
            args.validation_prompt = f.read().strip()
    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    if url.startswith("http://") or url.startswith("https://"):
        image = PIL.Image.open(requests.get(url, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw)
    else: # add option to locally load image
        image = PIL.Image.open(url)

    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image for inference."""
    # Convert from [-1, 1] to [0, 255]
    image = (tensor + 1.0) * 127.5
    image = image.clamp(0, 255).to(torch.uint8)
    # Convert from CHW to HWC
    image = image.permute(1, 2, 0)
    return PIL.Image.fromarray(image.cpu().numpy())

def extract_color_pixels(image: np.ndarray, lower_hue: int = 30, upper_hue: int = 90, saturation_threshold: int = 30, value_threshold: int = 20) -> np.ndarray:
  """
  Extract green pixels from the image with a given tolerance.
  
  Parameters:
  - image: Input image in BGR format.
  - lower_hue: Lower bound for the hue value for color to be extracted
  - upper_hue: Upper bound for the hue value forcolor to be extracted
  - saturation_threshold: Minimum saturation value to consider.
  - value_threshold: Minimum brightness value to consider.

  Returns:
  - green_mask: Mask of the same size as the image, with white pixels representing green areas.
  """
  # Convert BGR to HSV
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Define the lower and upper bounds for the color to extract
  lower_bound = np.array([lower_hue, saturation_threshold, value_threshold])
  upper_bound = np.array([upper_hue, 255, 255])

  # Create a mask for the color to extract
  color_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

  # Optional: Apply morphological operations to reduce noise
  apply_morphology = True
  if apply_morphology:
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

  return color_mask

  
def match_points(predicted, gt):
        predicted = list(predicted)
        gt = list(gt)
        matched_predicted = []
        unmatched_predicted = list(predicted)
        scores = []
        matched_count = 0
        for gt_pos in gt:
            if not predicted:
                scores.append(float('inf'))
                break
            distances = [(abs(gt_pos[0] - pred_pos[0]) + abs(gt_pos[1] - pred_pos[1]), pred_pos) for pred_pos in predicted]
            min_distance, nearest_pred_pos = min(distances, key=lambda x: x[0])
            scores.append(min_distance)
            matched_predicted.append(nearest_pred_pos)
            unmatched_predicted.remove(nearest_pred_pos)
            predicted.remove(nearest_pred_pos)
            matched_count += 1
        valid_scores = [s for s in scores if s != float('inf')]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else float('inf')
        
        matching_score = (len(predicted) - len(gt)) / len(gt) if gt else 0 # 0% best postive too many predictions, negative too few predictions

        return matched_predicted, unmatched_predicted, avg_score, matching_score


def extract_region_centers(mask: np.ndarray) -> list[tuple[int, int]]:
  """
  Extract the center (centroid) of each isolated region in the mask.
  
  Parameters:
  - mask: Binary mask where the regions of interest are white (255) and background is black (0).

  Returns:
  - centers: List of tuples representing the (x, y) coordinates of the centroids of each region.
  """
  # Find contours in the mask
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  centers = []
  for contour in contours:
    # Calculate moments for each contour
    M = cv2.moments(contour)
    
    if M['m00'] != 0:  # Avoid division by zero
      # Calculate the centroid
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])
      centers.append((cx, cy))
    else:
      # If the contour is too small, we might skip it or handle it differently
      pass

  return centers

def calculate_validation_score(predicted_img, ground_truth_img):
    """
    Calculate the validation score for both green (switches) and purple (routing points) pixels
    in the predicted image compared to the ground truth image using nearest neighbor matching with L1 distance.

    Parameters:
    - predicted_img: The image generated by the model.
    - ground_truth_img: The ground truth image.

    Returns:
    - Dictionary containing:
        - remaining_norm_count_green: Number of unmatched predicted green switches
        - average_valid_score_green: Average L1 distance for matched green switches
        - remaining_norm_count_purple: Number of unmatched predicted purple points
        - average_valid_score_purple: Average L1 distance for matched purple points
        - visualization_img: Image with matched switches/points in green/purple, unmatched in red
    """
    # Convert PIL Images to numpy arrays in BGR format for OpenCV
    predicted_np = cv2.cvtColor(np.array(predicted_img), cv2.COLOR_RGB2BGR)
    ground_truth_np = cv2.cvtColor(np.array(ground_truth_img), cv2.COLOR_RGB2BGR)

    # Green (switches)
    green_mask_prediction = extract_color_pixels(predicted_np)
    green_mask_ground_truth = extract_color_pixels(ground_truth_np)
    positions_predictions_green = extract_region_centers(green_mask_prediction)
    positions_ground_truth_green = extract_region_centers(green_mask_ground_truth)

    # Purple (routing points)
    purple_mask_prediction = extract_color_pixels(predicted_np, lower_hue=130, upper_hue=150)
    purple_mask_ground_truth = extract_color_pixels(ground_truth_np, lower_hue=130, upper_hue=150)
    positions_predictions_purple = extract_region_centers(purple_mask_prediction)
    positions_ground_truth_purple = extract_region_centers(purple_mask_ground_truth)

    # # print(len(positions_ground_truth_purple))
    # # print(len(positions_predictions_purple))

    # Match green points
    matched_green, unmatched_green, avg_score_green, matching_score_green = match_points(
        positions_predictions_green, positions_ground_truth_green
    )
    # Match purple points
    matched_purple, unmatched_purple, avg_score_purple, matching_score_purple = match_points(
        positions_predictions_purple, positions_ground_truth_purple
    )

    purple_color = (255, 0, 153)
    green_color = (0, 255, 0)
    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)


    # Visualization
    visualization_img = np.array(predicted_img).copy()
    visualization_img_bgr = cv2.cvtColor(visualization_img, cv2.COLOR_RGB2BGR)
    # Remove all green and purple pixels
    green_mask_viz = extract_color_pixels(visualization_img_bgr)
    purple_mask_viz = extract_color_pixels(visualization_img_bgr, lower_hue=130, upper_hue=150)
    visualization_img_bgr[green_mask_viz > 0] = [255, 255, 255]
    visualization_img_bgr[purple_mask_viz > 0] = [255, 255, 255]
    # Draw matched green switches in green
    radius = int(max(2, min(256, 256)*0.01))
    for pos in matched_green:
        cv2.circle(visualization_img_bgr, pos, radius=radius, color=green_color, thickness=-1) # attention here 256 needs to be changed in future!
    # Draw unmatched green switches in red
    for pos in unmatched_green:
        cv2.circle(visualization_img_bgr, pos, radius=radius, color=red_color, thickness=-1)
    # Draw matched purple points in purple (BGR: 255,0,255)
    for pos in matched_purple:
        cv2.circle(visualization_img_bgr, pos, radius=radius, color=purple_color, thickness=-1)
    # Draw unmatched purple points in yellow
    for pos in unmatched_purple:
        cv2.circle(visualization_img_bgr, pos, radius=radius, color=yellow_color, thickness=-1)
    # Convert back to RGB for PIL
    visualization_img_rgb = cv2.cvtColor(visualization_img_bgr, cv2.COLOR_BGR2RGB)
    visualization_pil = PIL.Image.fromarray(visualization_img_rgb)

    return {
        "matching_score_green": matching_score_green,
        "average_valid_score_green": avg_score_green,
        "matching_score_purple": matching_score_purple,
        "average_valid_score_purple": avg_score_purple,
        "visualization_img": visualization_pil
    }

# def _update_global_metrics_with_wandb(validation_scores, global_step, epoch):
#     """
#     Update global metrics by logging individual metric values directly to wandb for time-series plotting.
#     This creates proper WandB plots instead of using tracker.log().
#     """
#     if not validation_scores:
#         logger.warning("No validation scores to update global metrics")
#         return
        
#     try:
#         # Calculate aggregate metrics for this validation run
#         green_matching_scores = []
#         green_avg_scores = []
#         purple_matching_scores = []
#         purple_avg_scores = []
        
#         for score_metrics in validation_scores:
#             if score_metrics.get("matching_score_green") != float('inf') and not math.isnan(score_metrics.get("matching_score_green", float('nan'))):
#                 green_matching_scores.append(score_metrics.get("matching_score_green"))
#             if score_metrics.get("average_valid_score_green") != float('inf') and not math.isnan(score_metrics.get("average_valid_score_green", float('nan'))):
#                 green_avg_scores.append(score_metrics.get("average_valid_score_green"))
#             if score_metrics.get("matching_score_purple") != float('inf') and not math.isnan(score_metrics.get("matching_score_purple", float('nan'))):
#                 purple_matching_scores.append(score_metrics.get("matching_score_purple"))
#             if score_metrics.get("average_valid_score_purple") != float('inf') and not math.isnan(score_metrics.get("average_valid_score_purple", float('nan'))):
#                 purple_avg_scores.append(score_metrics.get("average_valid_score_purple"))
        
#         # Calculate means for this validation run
#         mean_green_matching = sum(green_matching_scores) / len(green_matching_scores) if green_matching_scores else float('nan')
#         mean_green_avg = sum(green_avg_scores) / len(green_avg_scores) if green_avg_scores else float('nan')
#         mean_purple_matching = sum(purple_matching_scores) / len(purple_matching_scores) if purple_matching_scores else float('nan')
#         mean_purple_avg = sum(purple_avg_scores) / len(purple_avg_scores) if purple_avg_scores else float('nan')
        
#         # Use global_step if available, otherwise use a sensible default
#         step_id = global_step if global_step is not None else 0
        
#         # Log individual metrics for time-series plotting using wandb.log directly
#         metrics_to_log = {
#             "validation/green_mean_matching_score": mean_green_matching,
#             "validation/green_mean_average_score": mean_green_avg,
#             "validation/purple_mean_matching_score": mean_purple_matching,
#             "validation/purple_mean_average_score": mean_purple_avg,
#         }
        
#         # Use wandb.log directly with explicit step for proper time-series plotting
#         wandb.log(metrics_to_log, step=step_id)
        
#         logger.info(f"Logged validation metrics to wandb at step {step_id}: Green (matching={mean_green_matching:.3f}, avg={mean_green_avg:.3f}, n={len(green_matching_scores)}), Purple (matching={mean_purple_matching:.3f}, avg={mean_purple_avg:.3f}, n={len(purple_matching_scores)})")
        
#     except Exception as e:
#         logger.error(f"Failed to log validation metrics to wandb: {e}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")


# ...existing code...
def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")

    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    if unet.conv_in.weight.shape[1] == 4:
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in
    else:
        # resuming from previous training with 4 channels
        logger.info("UNet.conv_in already has 8 channels — skipping reinitialization.")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None: # not used
            data_files["train"] = os.path.join(args.train_data_dir, "**")

        # print("Loading dataset, data files: ", data_files)
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    with accelerator.main_process_first():
        # Cast image columns to use HuggingFace's Image feature for memory efficiency
        logger.info("Setting up datasets with Image compression...")
        
        # Cast both image columns to use compressed storage
        dataset = dataset.cast_column(original_image_column, Image())
        dataset = dataset.cast_column(edited_image_column, Image())
        
        # Limit training samples if specified
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        else:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed)
        
        # Limit validation samples if specified
        if args.num_validation_images is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(args.num_validation_images))

        # Simple preprocessing that only handles tokenization - images processed on-the-fly
        def preprocess_fn(example):
            # Tokenize and convert to list (HuggingFace datasets handle lists better than tensors)
            input_ids = tokenize_captions([example[edit_prompt_column]])
            example["input_ids"] = input_ids.squeeze(0).tolist()  # Convert to list for HF compatibility
            return example

        # Apply minimal preprocessing (just tokenization)
        train_dataset = dataset["train"].map(preprocess_fn, remove_columns=[edit_prompt_column])
        val_dataset = dataset["validation"].map(preprocess_fn, remove_columns=[edit_prompt_column])
        
        logger.info(f"Using on-the-fly processing: {len(train_dataset)} train, {len(val_dataset)} val samples")


    # Create image transforms for just-in-time processing
    from torchvision import transforms as T

    def collate_fn(batch):
        """Just-in-time image decoding and processing collate function"""
        try:
            dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
            
            transform = transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
            
            # Decode and transform images on-the-fly
            original_pixel_values = torch.stack([
                transform(example[original_image_column]) for example in batch
            ])
            original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()

            edited_pixel_values = torch.stack([
                transform(example[edited_image_column]) for example in batch
            ])
            edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids_tensors = []
            for example in batch:
                ids = example["input_ids"]
                if isinstance(ids, list):
                    input_ids_tensors.append(torch.tensor(ids, dtype=torch.long))
                else:
                    input_ids_tensors.append(ids)
            input_ids = torch.stack(input_ids_tensors)
            
            return {
                "original_pixel_values": original_pixel_values.to(dtype),
                "edited_pixel_values": edited_pixel_values.to(dtype),
                "input_ids": input_ids,
            }
        except Exception as e:
            logger.error(f"Error in collate_fn: {e}")
            exit(1)

    # Simplified DataLoader creation
    max_workers = min(args.dataloader_num_workers, multiprocessing.cpu_count())
    
    # Use conservative batch sizes for validation to avoid CUDA OOM
    # Validation uses significantly more memory due to:
    # 1. Classifier-free guidance (2x embeddings)
    # 2. Image guidance (additional image embeddings)
    # 3. Multiple denoising steps with concatenated latents
    val_batch_size = 1  # Start with 1 to avoid OOM
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory > 20:  # More than 20GB GPU memory
            val_batch_size = 2
        elif gpu_memory > 16:  # More than 16GB GPU memory
            val_batch_size = 1  # Conservative for validation
        else:  # Less than 16GB GPU memory
            val_batch_size = 1

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=max_workers,
        pin_memory=True,
        persistent_workers=True if max_workers > 0 else False,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=val_batch_size,
        num_workers=max(1, max_workers // 2),
        pin_memory=True,
        persistent_workers=False,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Validation every {args.validation_batches} batches")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Setup validation flags once at the beginning
    use_enhanced_validation = (args.use_dataset_validation and val_dataloader is not None)
    use_fallback_validation = ((args.val_image_url is not None) 
                             and (args.validation_prompt is not None))

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        current_step_loss = 0.0  # Initialize loss tracking variable
        
        for step, batch in enumerate(train_dataloader):
            # print(f"DEBUG: Processing batch {step}, batch size: {batch['original_pixel_values'].shape[0]}")
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # Memory-optimized training step
            with accelerator.accumulate(unet):
                # print(f"DEBUG: Inside accelerator.accumulate at step {step}")
                
                # Clear cache before processing batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Convert to appropriate dtype early to save memory
                batch_original = batch["original_pixel_values"].to(weight_dtype)
                batch_edited = batch["edited_pixel_values"].to(weight_dtype)
                batch_input_ids = batch["input_ids"]
                
                # print(f"DEBUG: Batch shapes - original: {batch_original.shape}, edited: {batch_edited.shape}")
                
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(batch_edited).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch_input_ids)[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch_original).latent_dist.mode()
                
                # Clear batch tensors to save memory
                del batch_original, batch_edited

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://huggingface.co/papers/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # print(f"DEBUG: Loss calculated: {loss.item():.6f}")
                
                # Clear intermediate tensors
                del concatenated_noisy_latents, model_pred, target, noisy_latents, original_image_embeds
                del encoder_hidden_states, latents, noise

                # Backpropagate
                # print(f"DEBUG: Starting backward pass")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Clear cache after gradient step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # print(f"DEBUG: Gradient sync occurred at step {step}")
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss = avg_loss.item()
                current_step_loss = train_loss
                
                # print(f"DEBUG: Step {global_step} completed, loss: {train_loss:.6f}")
                logger.info(f"Step {global_step}: train_loss={train_loss:.4f}, step_loss={current_step_loss:.4f}, lr={lr_scheduler.get_last_lr()[0]:.6f}")

                # Log training metrics directly to wandb with explicit step for consistency with validation metrics
                # Only log once to avoid duplicate "train/loss" plots
                if accelerator.is_main_process and wandb is not None:
                    wandb.log({
                        "train/loss": train_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
                
                # Reset for next step
                train_loss = 0.0
                current_step_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Trigger validation during training based on global_step
                # Only run validation after the first validation interval, not at step 0
                # IMPORTANT: This check must be INSIDE the sync_gradients block to avoid duplicate validation calls
                if global_step > 0 and global_step % args.validation_batches == 0:
                    if use_enhanced_validation or use_fallback_validation:
                        if args.use_ema:
                            # Temporarily store and load EMA parameters for validation
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                        # Select the appropriate validation dataloader
                        validation_dataloader = val_dataloader if use_enhanced_validation else None

                        # Set models to evaluation mode
                        original_unet_training_mode = unet.training
                        original_vae_training_mode = vae.training
                        original_text_encoder_training_mode = text_encoder.training
                        
                        unet.eval()
                        vae.eval()
                        text_encoder.eval()

                        # Clean up memory to avoid issues
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        log_validation(
                            unet=unwrap_model(unet),
                            vae=unwrap_model(vae),
                            text_encoder=unwrap_model(text_encoder),
                            noise_scheduler=noise_scheduler,
                            args=args,
                            accelerator=accelerator,
                            generator=generator,
                            val_dataloader=validation_dataloader,
                            tokenizer=tokenizer,
                            epoch=epoch,
                            global_step=global_step,
                            weight_dtype=weight_dtype
                        )
                        
                        # Restore original training modes
                        unet.train(original_unet_training_mode)
                        vae.train(original_vae_training_mode) 
                        text_encoder.train(original_text_encoder_training_mode)

                        if args.use_ema:
                            # Restore original UNet parameters
                            ema_unet.restore(unet.parameters())

                        # Clean up memory to avoid issues
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # Get the last loss value for logging (use stored value or default)
            step_loss_for_display = current_step_loss if 'current_step_loss' in locals() and current_step_loss is not None else 0.0
            logs = {"step_loss": step_loss_for_display, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # finished training
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
            safety_checker=None if args.disable_safety_checker else "default",
            requires_safety_checker=not args.disable_safety_checker,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        # Final validation
        use_enhanced_validation = (args.use_dataset_validation and val_dataloader is not None)
        use_fallback_validation = ((args.val_image_url is not None) 
                                 and (args.validation_prompt is not None))
        
        if use_enhanced_validation or use_fallback_validation:
            validation_dataloader = val_dataloader if use_enhanced_validation else None
            log_validation(
                unet=unwrap_model(unet),
                vae=unwrap_model(vae),
                text_encoder=unwrap_model(text_encoder),
                noise_scheduler=noise_scheduler,
                args=args,
                accelerator=accelerator,
                generator=generator,
                val_dataloader=validation_dataloader,  # Pass validation dataloader if enhanced validation
                tokenizer=tokenizer,   # Pass tokenizer
                epoch=None,  # Final validation, no specific epoch
                global_step=global_step,  # Pass current global step for logging
                weight_dtype=weight_dtype
            )
    accelerator.end_training()


def run_diffusion_inference_batch(unet, vae, text_encoder, noise_scheduler, original_images_batch, prompts_batch, tokenizer, device, generator, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7):
    """
    Memory-optimized batch diffusion inference using model components directly.
    
    Args:
        unet: The trained UNet model
        vae: The VAE model
        text_encoder: The text encoder model
        noise_scheduler: The noise scheduler
        original_images_batch: Batch of original images as tensors [B, 3, H, W]
        prompts_batch: List of prompts for each image
        tokenizer: The tokenizer
        device: The device to run inference on
        generator: Random generator for reproducibility
        num_inference_steps: Number of denoising steps
        image_guidance_scale: Image guidance scale
        guidance_scale: Text guidance scale
    
    Returns:
        List of PIL Images
    """
    batch_size = original_images_batch.shape[0]
    
    # Force cleanup before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Tokenize prompts
    text_inputs = tokenizer(
        prompts_batch,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids)[0]
    
    # Prepare unconditional embeddings for classifier-free guidance
    uncond_tokens = [""] * batch_size
    uncond_inputs = tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    uncond_input_ids = uncond_inputs.input_ids.to(device)
    
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input_ids)[0]
    
    # Concatenate for classifier-free guidance
    text_embeddings_combined = torch.cat([uncond_embeddings, text_embeddings])
    
    # Clear intermediate tensors
    del text_input_ids, uncond_input_ids, text_embeddings, uncond_embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Encode original images to latent space
    with torch.no_grad():
        original_image_embeds = vae.encode(original_images_batch).latent_dist.mode()
    
    # Prepare latents for generation
    latents_shape = (batch_size, unet.config.in_channels // 2, original_image_embeds.shape[2], original_image_embeds.shape[3])
    latents = torch.randn(latents_shape, generator=generator, device=device, dtype=original_images_batch.dtype)
    latents = latents * noise_scheduler.init_noise_sigma
    
    # Set timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps
    
    # Prepare image embeddings for guidance
    original_image_embeds_combined = torch.cat([original_image_embeds] * 2)
    
    # Clear intermediate tensors
    del original_image_embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # Expand latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        # Concatenate with image embeddings
        latent_model_input = torch.cat([latent_model_input, original_image_embeds_combined], dim=1)
        
        # Predict noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_combined, return_dict=False)[0]
        
        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Apply image guidance if needed
        if image_guidance_scale != 1.0:
            # Simple image guidance implementation
            noise_pred = noise_pred_uncond + image_guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Compute previous noisy sample
        latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Clear intermediate tensors
        del latent_model_input, noise_pred, noise_pred_uncond, noise_pred_text
        
        # Periodic cleanup during denoising
        if i % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clear embeddings before decoding
    del text_embeddings_combined, original_image_embeds_combined
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Decode latents to images
    with torch.no_grad():
        images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    
    # Clear latents
    del latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert to PIL images
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    pil_images = [PIL.Image.fromarray(image) for image in images]
    
    # Final cleanup
    del images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return pil_images


if __name__ == "__main__":
    main()
