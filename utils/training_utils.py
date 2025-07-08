"""
Training utilities for InstructPix2Pix.
Contains helper functions for the training loop and checkpoint management.
"""

import gc
import os
import logging
import shutil
import torch
import math

logger = logging.getLogger(__name__)

def compute_training_steps(args, train_dataloader, accelerator):
    """
    Calculate training steps and configure learning rate scheduler.
    
    Args:
        args: Training arguments
        train_dataloader: Training data loader
        accelerator: Accelerator instance
        
    Returns:
        Dictionary with step configuration and scheduler
    """
    from diffusers.optimization import get_scheduler
    
    # Calculate steps for scheduler
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    
    if args.max_train_steps is None:
        # Calculate based on epochs
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        # Use explicitly provided max steps
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes
    
    # Create the scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=args.optimizer,  # Passed in from main script
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    
    # Recalculate after accelerator setup
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    # Update max_train_steps if not explicitly set
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        
        # Check for inconsistencies in scheduler vs actual steps
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    
    # Recalculate number of epochs based on max_train_steps
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    return {
        "lr_scheduler": lr_scheduler,
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "num_training_steps": num_training_steps_for_scheduler
    }


def setup_checkpoint_handlers(args, accelerator, unet, ema_unet=None):
    """
    Set up custom saving and loading hooks for checkpointing.
    
    Args:
        args: Training arguments
        accelerator: Accelerator instance
        unet: UNet model
        ema_unet: EMA UNet model (optional)
    """
    # Only set up custom handlers for accelerate >= 0.16.0
    try:
        import accelerate
        from packaging import version
        if version.parse(accelerate.__version__) < version.parse("0.16.0"):
            logger.warning("Custom checkpoint handlers require accelerate >= 0.16.0. Skipping.")
            return
    except (ImportError, AttributeError):
        logger.warning("Could not determine accelerate version. Skipping custom checkpoint handlers.")
        return
    
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            logger.info(f"Starting checkpoint save to {output_dir}")
            
            # Force memory cleanup before saving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"CHECKPOINT - Before save: GPU={memory_before:.2f}GB")
            
            try:
                if args.use_ema and ema_unet is not None:
                    logger.info("Saving EMA model...")
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                    
                    # Force cleanup after EMA save
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                for i, model in enumerate(models):
                    logger.info(f"Saving model {i+1}/{len(models)}...")
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()
                    
                    # Force cleanup after each model save
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Final aggressive cleanup after all saves
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    memory_after = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"CHECKPOINT - After save: GPU={memory_after:.2f}GB")
                
                logger.info(f"Checkpoint save completed successfully")
                
            except Exception as e:
                logger.error(f"Error during checkpoint save: {e}")
                # Emergency cleanup on error
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

    def load_model_hook(models, input_dir):
        from diffusers import UNet2DConditionModel
        from diffusers.training_utils import EMAModel
        
        logger.info(f"Loading checkpoint from {input_dir}")
        
        if args.use_ema and ema_unet is not None:
            logger.info("Loading EMA model...")
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model
            
            # Cleanup after EMA load
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for i in range(len(models)):
            logger.info(f"Loading model {i+1}/{len(models)}...")
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
            
            # Cleanup after each model load
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final cleanup after all loads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Checkpoint load completed with memory cleanup")

    # Register the hooks with accelerator
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def manage_checkpoints(args, output_dir, global_step):
    """
    Manage checkpoints based on total limit and save current checkpoint.
    
    Args:
        args: Training arguments
        output_dir: Output directory
        global_step: Current global step
        
    Returns:
        save_path: Path where checkpoint was saved
    """
    # Check if we need to enforce total checkpoint limit
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # Before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint_path = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint_path)

    # Create checkpoint path
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    
    # Force memory cleanup before checkpoint save
    logger.info(f"Preparing to save checkpoint at step {global_step}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    return save_path


def resume_from_checkpoint(args, accelerator):
    """
    Handle resuming from a checkpoint.
    
    Args:
        args: Training arguments
        accelerator: Accelerator instance
        
    Returns:
        Dictionary with resume state information
    """
    resume_state = {
        "global_step": 0,
        "first_epoch": 0,
        "resume_step": 0
    }
    
    # Check if we need to resume from checkpoint
    if not args.resume_from_checkpoint:
        return resume_state
        
    # Determine checkpoint path
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
        args.resume_from_checkpoint = None
        return resume_state
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        checkpoint_global_step = int(path.split("-")[1])
        
        # Handle case where checkpoint global_step exceeds max_train_steps
        # This can happen when resuming with different epoch/dataset settings
        if checkpoint_global_step >= args.max_train_steps:
            logger.warning(
                f"Checkpoint global_step ({checkpoint_global_step}) >= max_train_steps ({args.max_train_steps}). "
                f"This usually means the training configuration has changed."
            )
            
            if args.extend_training_on_resume:
                # Option 2: Extend max_train_steps to continue training from checkpoint
                num_update_steps_per_epoch = math.ceil(
                    args.dataloader_length / args.gradient_accumulation_steps
                )
                args.max_train_steps = checkpoint_global_step + (args.num_train_epochs * num_update_steps_per_epoch)
                logger.info(f"Extended max_train_steps to {args.max_train_steps} to continue training from checkpoint")
                accelerator.load_state(os.path.join(args.output_dir, path))
                resume_state = {
                    "global_step": checkpoint_global_step,
                    "first_epoch": checkpoint_global_step // num_update_steps_per_epoch,
                    "resume_step": (checkpoint_global_step * args.gradient_accumulation_steps) % 
                                  (num_update_steps_per_epoch * args.gradient_accumulation_steps)
                }
            else:
                # Option 1: Load the checkpoint but start fresh with global_step=0
                logger.info("Loading model weights from checkpoint but resetting global_step to 0 for continued training")
                logger.info("Use --extend_training_on_resume to continue from checkpoint step instead")
                accelerator.load_state(os.path.join(args.output_dir, path))
                resume_state = {
                    "global_step": 0,
                    "first_epoch": 0,
                    "resume_step": 0
                }
            
        else:
            # Normal checkpoint resumption
            accelerator.load_state(os.path.join(args.output_dir, path))
            num_update_steps_per_epoch = math.ceil(
                args.dataloader_length / args.gradient_accumulation_steps
            )
            resume_state = {
                "global_step": checkpoint_global_step,
                "first_epoch": checkpoint_global_step // num_update_steps_per_epoch,
                "resume_step": (checkpoint_global_step * args.gradient_accumulation_steps) % 
                              (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            }
            
    return resume_state
