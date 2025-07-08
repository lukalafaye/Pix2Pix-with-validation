"""
Diffusion inference utilities for InstructPix2Pix training.
Contains the batch inference function for memory-optimized image generation.
"""

import os
import torch
import psutil
import PIL
from accelerate.logging import get_logger
from diffusers.utils.torch_utils import is_compiled_module

logger = get_logger(__name__, log_level="INFO")

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


def run_diffusion_inference_batch(unet, vae, text_encoder, noise_scheduler, original_images_batch, 
                                 prompts_batch, tokenizer, device, generator, num_inference_steps=20, 
                                 image_guidance_scale=1.5, guidance_scale=7, input_is_latents=False,
                                 return_latents=False, pre_tokenized_ids=None):
    """
    Memory-optimized batch diffusion inference using model components directly.
    
    Args:
        unet: The trained UNet model
        vae: The VAE model
        text_encoder: The text encoder model
        noise_scheduler: The noise scheduler
        original_images_batch: Batch of original images as tensors [B, 3, H, W] or latents if input_is_latents=True
        prompts_batch: List of prompts for each image (not used if pre_tokenized_ids is provided)
        tokenizer: The tokenizer
        device: The device to run inference on
        generator: Random generator for reproducibility
        num_inference_steps: Number of denoising steps
        image_guidance_scale: Image guidance scale
        guidance_scale: Text guidance scale
        input_is_latents: If True, assumes original_images_batch are already VAE latents (skips encoding)
        return_latents: If True, returns the latent representations instead of decoded PIL images
        pre_tokenized_ids: Optional pre-tokenized input IDs (batch_size, seq_len), skips tokenization step
    
    Returns:
        List of PIL Images or latent tensors if return_latents is True
    """
    batch_size = original_images_batch.shape[0]
    
    # Log memory before inference starts
    log_memory_usage(f"Inference start - batch_size={batch_size}", reset_peak=True)
    
    # Force cleanup before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get text input IDs - either use provided tokens or tokenize prompts
    if pre_tokenized_ids is not None:
        # Use pre-tokenized IDs directly
        logger.info("Using pre-tokenized input IDs (skipping tokenization)")
        text_input_ids = pre_tokenized_ids.to(device)
    else:
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
        log_memory_usage("Inference - Before text embedding")
        text_embeddings = text_encoder(text_input_ids)[0]
        log_memory_usage("Inference - After text embedding")
    
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
    
    # Get original image latents - either encode or use directly
    with torch.no_grad():
        if input_is_latents:
            # Use provided latents directly
            original_image_embeds = original_images_batch
            logger.info("Using provided latents directly (skipping VAE encoding)")
        else:
            # Encode original images to latent space
            log_memory_usage("Inference - Before VAE encoding")
            original_image_embeds = vae.encode(original_images_batch).latent_dist.mode()
            log_memory_usage("Inference - After VAE encoding")
    
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
            # Log memory usage during first denoising step and middle step
            if i == 0 or i == num_inference_steps // 2:
                log_memory_usage(f"Inference step {i}/{num_inference_steps} - Before UNet")
            
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_combined, return_dict=False)[0]
            
            # Log memory after UNet pass for critical steps
            if i == 0 or i == num_inference_steps // 2:
                log_memory_usage(f"Inference step {i}/{num_inference_steps} - After UNet")
        
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
    
    # Return latents if requested (skip decoding)
    if return_latents:
        log_memory_usage("Inference completed - returning latents")
        return latents
    
    # Decode latents to images
    with torch.no_grad():
        log_memory_usage("Inference - Before VAE decoding")
        images = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        log_memory_usage("Inference - After VAE decoding")
    
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
        torch.cuda.synchronize()
    
    # Log memory at the end of inference
    log_memory_usage("Inference completed")
    
    return pil_images
