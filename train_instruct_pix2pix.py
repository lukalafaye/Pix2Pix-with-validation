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

"""
InstructPix2Pix Training Script

This script fine-tunes Stable Diffusion for instruction-based image editing using the 
InstructPix2Pix technique. The code has been refactored into a modular structure with
utility functions moved to the utils directory for better organization and maintainability.

The refactored structure includes:
- utils/image_processing.py: Image conversion and manipulation utilities
- utils/validation.py: Validation metrics and testing
- utils/inference.py: Model inference logic
- utils/logging_utils.py: Logging and visualization helpers
- utils/data_utils.py: Dataset preprocessing and dataloader creation
- utils/model_utils.py: Model setup, configuration, and optimization
- utils/training_utils.py: Training step calculation, checkpointing, and resuming

This modularization aims to improve code readability, maintainability, and reusability
while preserving the core functionality of the InstructPix2Pix training process.
"""

# Standard library imports
import argparse
import gc
import logging
import math
import multiprocessing
import os
import psutil
import shutil
import time
from pathlib import Path
from contextlib import nullcontext

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import cv2
from tqdm.auto import tqdm
from packaging import version
from joblib import Parallel, delayed

# HuggingFace imports
import accelerate
import datasets
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import DatasetDict, Image, load_dataset
from huggingface_hub import create_repo, upload_folder, HfFolder
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from diffusers import (
    AutoencoderKL, DDPMScheduler, 
    StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# Import custom utility modules - consolidated imports
from utils import (
    # Image processing utilities
    convert_to_np, download_image, tensor_to_pil,
    
    # Validation and inference
    calculate_validation_score, run_diffusion_inference_batch,
    
    # Logging and validation
    log_validation,
    
    # Data processing
    create_dataloaders, preprocess_dataset,
    
    # Model and training utilities
    setup_models, setup_optimizer, unwrap_model,
    compute_training_steps, setup_checkpoint_handlers,
    manage_checkpoints, resume_from_checkpoint
)
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
    parser.add_argument(
        "--extend_training_on_resume",
        action="store_true",
        help=(
            "When resuming from a checkpoint where global_step > max_train_steps, extend max_train_steps "
            "to continue training instead of resetting global_step to 0. Use this to continue training "
            "beyond the originally planned epochs."
        ),
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


# Image processing functions have been moved to utils/image_processing.py

# All utility functions (convert_to_np, download_image, tensor_to_pil, extract_color_pixels, 
# extract_region_centers, match_points, calculate_validation_score, run_diffusion_inference_batch)
# have been moved to their respective utility modules in the utils directory.


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
def log_memory_usage(label="Memory Usage", accelerator=None, reset_peak=False):
    """
    Log GPU and RAM memory usage
    
    Args:
        label: Label for the log entry
        accelerator: Optional accelerator instance to check if this is the main process
        reset_peak: If True, reset the peak memory tracking after logging
    """
    if accelerator and not accelerator.is_local_main_process:
        return  # Only log on main process
    
    # Get RAM usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
    
    log_message = f"{label}: RAM={ram_usage:.2f}GB"
    
    # Get GPU usage if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all CUDA operations to finish
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)  # GB
        gpu_max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # GB
        
        log_message += f", GPU Allocated={gpu_allocated:.2f}GB, Reserved={gpu_reserved:.2f}GB, Peak={gpu_max_memory:.2f}GB"
        
        # Reset peak memory stats if requested
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
            log_message += " (peak reset)"
        
    logger.info(log_message)


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

    # Set up all models using our utility function
    # This handles model loading, UNet adaptation, freezing, and optimizations
    models = setup_models(args, accelerator)
    
    # Unpack the models
    noise_scheduler = models["noise_scheduler"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    unet = models["unet"]
    ema_unet = models["ema_unet"]
    
    # Define a local helper for model unwrapping
    def get_unwrapped_model(model):
        return unwrap_model(model, accelerator)

    # Set up checkpoint handlers using our utility function
    setup_checkpoint_handlers(args, accelerator, unet, ema_unet)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Set number of processes for scaling (if needed)
    args.num_processes = accelerator.num_processes
    
    # Set up optimizer using our utility function
    optimizer = setup_optimizer(unet, args)

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

    # Preprocessing the datasets using our utility function
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

        # Use our utility function to preprocess the dataset
        train_dataset, val_dataset = preprocess_dataset(
            dataset=dataset, 
            args=args,
            tokenizer=tokenizer
        )


    # Create dataloaders using our utility function
    # This handles collate_fn creation and all dataloader configuration internally
    logger.info("Creating optimized dataloaders for training and validation...")
    
    # Store original image and edited image columns in args for the data utility functions
    args.original_image_column = original_image_column
    args.edited_image_column = edited_image_column
    
    # Make sure dataloader_num_workers is defined in args
    if not hasattr(args, 'dataloader_num_workers'):
        args.dataloader_num_workers = min(8, multiprocessing.cpu_count())
        logger.info(f"Setting default dataloader_num_workers to {args.dataloader_num_workers}")
    
    # Create dataloaders with optimized settings
    train_dataloader, val_dataloader = create_dataloaders(
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        multiprocessing=multiprocessing
    )

    # Configure training steps and scheduler using our utility function
    # Store dataloader length for resume calculation
    args.dataloader_length = len(train_dataloader)
    args.optimizer = optimizer  # Pass optimizer to the utility function
    
    # Calculate training steps and create scheduler
    training_config = compute_training_steps(args, train_dataloader, accelerator)
    lr_scheduler = training_config["lr_scheduler"]

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
    # Get updated values from training_config
    num_update_steps_per_epoch = training_config["num_update_steps_per_epoch"]
    
    # Use the calculated values from our utility function
    # These have already been properly handled in compute_training_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    # Initialize wandb variable first
    wandb_module = None
    
    # Check if wandb should be initialized here
    if is_wandb_available():
        import wandb as wandb_module
        if args.report_to == "wandb" and accelerator.is_main_process:
            # Only initialize if not already initialized
            if not hasattr(wandb_module, "run") or wandb_module.run is None:
                logger.info("Initializing wandb for main process")
                project_name = "instruct-pix2pix"
                run_name = os.path.basename(args.output_dir)
                try:
                    wandb_module.init(project=project_name, name=run_name)
                    logger.info(f"Wandb initialized with run: {wandb_module.run.name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize wandb: {e}")
                    wandb_module = None

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # This section has been moved to the correct location before the progress bar initialization

    # Initialize step variables
    global_step = 0
    first_epoch = 0
    resume_step = 0
    
    # Handle resuming from a checkpoint using our utility function
    if args.resume_from_checkpoint:
        resume_state = resume_from_checkpoint(args, accelerator)
        global_step = resume_state["global_step"]
        first_epoch = resume_state["first_epoch"]
        resume_step = resume_state["resume_step"]
        logger.info(f"Resuming from checkpoint: global_step={global_step}, first_epoch={first_epoch}, resume_step={resume_step}")

    # Validate training configuration
    if global_step >= args.max_train_steps:
        logger.error(
            f"ERROR: Starting global_step ({global_step}) >= max_train_steps ({args.max_train_steps}). "
            f"Training would exit immediately. Please check your configuration."
        )
        # Don't exit here - let the fix above handle it gracefully

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Validation every {args.validation_batches} batches")
    logger.info(f"  Starting global_step = {global_step}")
    logger.info(f"  Starting first_epoch = {first_epoch}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Setup validation flags once at the beginning
    # Enhanced validation uses the validation dataset with the validation dataloader
    use_enhanced_validation = (getattr(args, 'use_dataset_validation', False) and val_dataloader is not None)
    
    # Fallback validation uses a single image URL and prompt for validation
    use_fallback_validation = ((args.val_image_url is not None) 
                             and (args.validation_prompt is not None))
                             
    logger.info(f"Validation setup: Enhanced validation: {use_enhanced_validation}, Fallback validation: {use_fallback_validation}")

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
                # Log memory usage before training step
                if step % 50 == 0:  # Only log every 50 steps to avoid log spam
                    log_memory_usage(f"Step {step} - Before training", accelerator, reset_peak=True)
                
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
                    # Create empty token directly
                    empty_tokens = tokenizer(
                        [""], max_length=tokenizer.model_max_length, 
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(accelerator.device)
                    null_conditioning = text_encoder(empty_tokens)[0]
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
                    
                # Log memory usage after training step
                if step % 50 == 0:  # Only log every 50 steps to avoid log spam
                    log_memory_usage(f"Step {step} - After training", accelerator)

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
                if accelerator.is_main_process and wandb_module is not None:
                    wandb_module.log({
                        "train/loss": train_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
                
                # Reset for next step
                train_loss = 0.0
                current_step_loss = 0.0

                # Save checkpoint at regular intervals
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Use our utility function to manage checkpoints
                        save_path = manage_checkpoints(
                            args=args,
                            output_dir=args.output_dir,
                            global_step=global_step
                        )
                        
                        # Save with accelerator
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

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
                            
                        # Log memory usage before validation
                        log_memory_usage(f"Global step {global_step} - Before validation", accelerator, reset_peak=True)
                        
                        log_validation(
                            unet=get_unwrapped_model(unet),
                            vae=get_unwrapped_model(vae),
                            text_encoder=get_unwrapped_model(text_encoder),
                            noise_scheduler=noise_scheduler,
                            args=args,
                            accelerator=accelerator,
                            generator=generator,
                            val_dataloader=validation_dataloader,
                            tokenizer=tokenizer,
                            epoch=epoch,
                            global_step=global_step,
                            weight_dtype=weight_dtype,
                            wandb=wandb_module  # Pass wandb module
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
                            
                        # Log memory usage after validation
                        log_memory_usage(f"Global step {global_step} - After validation", accelerator)

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
            text_encoder=get_unwrapped_model(text_encoder),
            vae=get_unwrapped_model(vae),
            unet=get_unwrapped_model(unet),
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
        use_enhanced_validation = (getattr(args, 'use_dataset_validation', False) and val_dataloader is not None)
        use_fallback_validation = ((args.val_image_url is not None) 
                                 and (args.validation_prompt is not None))
        
        if use_enhanced_validation or use_fallback_validation:
            validation_dataloader = val_dataloader if use_enhanced_validation else None
            
            # Log memory before final validation
            log_memory_usage("Final validation - Before", accelerator)
            
            log_validation(
                unet=get_unwrapped_model(unet),
                vae=get_unwrapped_model(vae),
                text_encoder=get_unwrapped_model(text_encoder),
                noise_scheduler=noise_scheduler,
                args=args,
                accelerator=accelerator,
                generator=generator,
                val_dataloader=validation_dataloader,  # Pass validation dataloader if enhanced validation
                tokenizer=tokenizer,   # Pass tokenizer
                epoch=None,  # Final validation, no specific epoch
                global_step=global_step,  # Pass current global step for logging
                weight_dtype=weight_dtype,
                wandb=wandb_module  # Pass wandb module
            )
            
            # Log memory after final validation
            log_memory_usage("Final validation - After", accelerator)
    accelerator.end_training()


# run_diffusion_inference_batch has been moved to utils/inference.py



if __name__ == "__main__":
    main()
