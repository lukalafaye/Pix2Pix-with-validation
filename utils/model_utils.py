"""
Model utilities for InstructPix2Pix training.
Contains functions for model setup, initialization and configuration.
"""

import logging
import torch
import torch.nn as nn
from accelerate.utils import set_seed
from diffusers.utils.torch_utils import is_compiled_module

logger = logging.getLogger(__name__)

def setup_models(args, accelerator):
    """
    Set up and initialize all required models for training.
    
    Args:
        args: Training arguments
        accelerator: Accelerator instance
        
    Returns:
        Dictionary containing all model components and configuration
    """
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.training_utils import EMAModel
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
        
    logger.info("Initializing models...")
    
    # Load scheduler, tokenizer and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="tokenizer", 
        revision=args.revision
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder", 
        revision=args.revision, 
        variant=args.variant
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="vae", 
        revision=args.revision, 
        variant=args.variant
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=args.non_ema_revision
    )
    
    # InstructPix2Pix uses 8 channels in the first UNet layer (instead of 4)
    # to accommodate the additional image for conditioning
    logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)
    
    if unet.conv_in.weight.shape[1] == 4:
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, 
                out_channels, 
                unet.conv_in.kernel_size, 
                unet.conv_in.stride, 
                unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in
    else:
        # resuming from previous training with 8 channels
        logger.info("UNet.conv_in already has 8 channels — skipping reinitialization.")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Create EMA for the unet if requested
    ema_unet = None
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        
    # Enable memory optimizations if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        
    if args.enable_xformers_memory_efficient_attention:
        try:
            import xformers
            from diffusers.utils.import_utils import is_xformers_available
            
            if is_xformers_available():
                import xformers
                xformers_version = xformers.__version__
                unet.enable_xformers_memory_efficient_attention()
                logger.info(f"Using xFormers version {xformers_version} for memory-efficient attention")
            else:
                raise ValueError("xformers is not available")
        except (ImportError, ValueError) as e:
            logger.error(f"Error enabling xformers: {e}")
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # Enable TF32 for faster training on Ampere GPUs if requested
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("Enabled TF32 for faster training on Ampere GPUs")
    
    # Return all model components
    return {
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "ema_unet": ema_unet
    }


def setup_optimizer(unet, args):
    """
    Configure the optimizer for training.
    
    Args:
        unet: The UNet model being trained
        args: Training arguments
        
    Returns:
        optimizer: Configured optimizer
    """
    # Scale learning rate based on batch size and processes if requested
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * 
            args.train_batch_size * args.num_processes
        )
        logger.info(f"Scaled learning rate to: {args.learning_rate}")
    
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer")
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. "
                "You can do so by running `pip install bitsandbytes`"
            )
    else:
        optimizer_cls = torch.optim.AdamW
        
    # Create and return the optimizer
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    logger.info(f"Initialized {optimizer_cls.__name__} optimizer with lr={args.learning_rate}")
    
    return optimizer


def unwrap_model(model, accelerator):
    """
    Unwrap a model from its accelerator and compiled state.
    
    Args:
        model: The model to unwrap
        accelerator: The accelerator instance
        
    Returns:
        unwrapped_model: The unwrapped model
    """
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model
