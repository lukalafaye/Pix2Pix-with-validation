"""
Utility modules for InstructPix2Pix training.
This package provides modular functionality for different aspects of training.

Modules:
- image_processing: Functions for image manipulation and conversion
- validation: Validation scoring and metrics
- inference: Model inference utilities
- logging_utils: Logging and validation orchestration
- data_utils: Dataset processing and dataloader creation
- model_utils: Model setup and configuration
- training_utils: Training loop helpers and checkpoint management
- auxiliary_loss: Auxiliary loss functions for targeted training
"""

# Import key functions for easier access
from .image_processing import convert_to_np,download_image,tensor_to_pil,extract_color_pixels,extract_region_centers,match_points,decode_latents_to_opencv_images,is_image_black
from .validation import calculate_validation_score
from .inference import log_memory_usage,run_diffusion_inference_batch

from .logging_utils import ensure_wandb_init,log_memory_usage,log_validation

from .data_utils import create_dataloaders, preprocess_dataset
from .auxiliary_loss import (
    dice_loss, rgb_to_hsv_torch, hue_mask, dice_loss, compute_auxiliary_losses
)

# Note: model_utils and training_utils imports removed as those modules don't exist yet
# Add them back when those modules are created:
from .model_utils import setup_models, setup_optimizer, unwrap_model
from .training_utils import (
compute_training_steps, setup_checkpoint_handlers, manage_checkpoints, resume_from_checkpoint
)
