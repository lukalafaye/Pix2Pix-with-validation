"""
Logging and validation utilities for InstructPix2Pix training.
Contains the main validation function that orchestrates the entire validation process.
"""

import os
import torch
import psutil
from contextlib import nullcontext
from joblib import Parallel, delayed
from accelerate.logging import get_logger
from .image_processing import tensor_to_pil
from .validation import calculate_validation_score
from .inference import run_diffusion_inference_batch

logger = get_logger(__name__, log_level="INFO")


def ensure_wandb_init(wandb_module=None, project_name="instruct-pix2pix", run_name=None):
    """
    Ensure that wandb is properly initialized if it's available
    
    Args:
        wandb_module: The wandb module (if already imported)
        project_name: The name of the wandb project
        run_name: Optional name for this run
        
    Returns:
        The wandb module if available and initialized, None otherwise
    """
    try:
        if wandb_module is None:
            try:
                import wandb
                wandb_module = wandb
            except ImportError:
                logger.warning("Failed to import wandb, will not log to wandb")
                return None
        
        # Check if wandb is already initialized
        try:
            if not hasattr(wandb_module, "run") or wandb_module.run is None:
                logger.info(f"Initializing wandb with project={project_name}, run_name={run_name}")
                wandb_module.init(project=project_name, name=run_name)
                logger.info(f"Successfully initialized wandb: {wandb_module.run.name}")
            else:
                logger.info(f"WandB already initialized with run: {wandb_module.run.name}")
        except AttributeError:
            logger.warning("Could not verify WandB initialization status - attempting to initialize")
            try:
                wandb_module.init(project=project_name, name=run_name)
                logger.info(f"Successfully initialized wandb: {wandb_module.run.name}")
            except Exception as init_error:
                logger.warning(f"WandB initialization failed: {init_error}")
                
        return wandb_module
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {str(e)}")
        return None


def log_memory_usage(label="Memory Usage", reset_peak=False):
    """
    Log GPU and RAM memory usage
    
    Args:
        label: Label for the log entry
        reset_peak: If True, reset the peak memory tracking after logging
    """
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


def log_validation(unet, vae, text_encoder, noise_scheduler, args, accelerator, generator,
                  val_dataloader=None, tokenizer=None, epoch=None, global_step=None, 
                  weight_dtype=torch.float32, wandb=None):
    """
    Perform validation during training with comprehensive scoring and logging.
    
    Args:
        unet: The UNet model being trained
        vae: The VAE model
        text_encoder: The text encoder model
        noise_scheduler: The noise scheduler
        args: Training arguments
        accelerator: The accelerator object
        generator: Random generator for reproducibility
        val_dataloader: Validation dataloader (optional)
        tokenizer: The tokenizer
        epoch: Current epoch number
        global_step: Current global step
        weight_dtype: Data type for model weights
        wandb: WandB module for logging
    """
    # Ensure wandb is initialized if specified in args
    if accelerator.is_main_process and args.report_to == "wandb":
        wandb = ensure_wandb_init(wandb, project_name="instruct-pix2pix", 
                                 run_name=f"{os.path.basename(args.output_dir)}")
        if wandb is None:
            logger.warning("WandB was requested but couldn't be initialized. Check your wandb installation.")
    
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

    # Use validation dataloader for comprehensive validation
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
                
                # Process batch - store both tokenized IDs and decoded prompts for debugging
                batch_prompts = []
                for i in range(batch_size):
                    # Store decoded prompts only for logging/debugging purposes
                    input_ids = batch["input_ids"][i]
                    prompt = tokenizer.decode(input_ids, skip_special_tokens=True).strip()
                    batch_prompts.append(prompt)
                
                # Move batch tensors to CPU immediately and delete originals
                original_pixel_values_cpu = batch["original_pixel_values"].cpu()
                edited_pixel_values_cpu = batch["edited_pixel_values"].cpu()
                input_ids_cpu = batch["input_ids"].cpu()  # Keep tokenized IDs for direct use in inference
                
                logger.debug(f"Using pre-tokenized IDs directly, decoded prompt examples: {batch_prompts[:2]}")
                
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
                    
                    # Log memory before batch inference
                    log_memory_usage(f"Batch {batch_idx} - Before inference", reset_peak=True)
                    
                    # Option 1: Use pre-tokenized IDs directly (avoiding decode-encode cycle)
                    # Use CPU tensors moved back to GPU only for inference
                    predicted_batch = run_diffusion_inference_batch(
                        unet, vae, text_encoder, noise_scheduler,
                        original_pixel_values_cpu.to(accelerator.device, dtype=weight_dtype),
                        None,  # Not using text prompts when providing pre-tokenized IDs
                        tokenizer,
                        accelerator.device,
                        generator,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7,
                        input_is_latents=False,
                        return_latents=False,
                        pre_tokenized_ids=input_ids_cpu.to(accelerator.device)  # Use pre-tokenized IDs directly
                    )
                    
                    # Images are already PIL Images on CPU from run_diffusion_inference_batch
                    predicted_batch_cpu = [img.copy() for img in predicted_batch]  # Ensure CPU copy

                    del predicted_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Log memory after batch inference
                    log_memory_usage(f"Batch {batch_idx} - After inference")

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
                                wandb.Image(filtered_predicted_images[i]),
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
        logger.warning("No validation dataloader provided, skipping validation")
    
    # Final memory cleanup at end of validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Final memory debug info
        memory_after_validation = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"MEMORY DEBUG - After complete validation: Allocated={memory_after_validation:.2f}GB")
