"""
Auxiliary loss functions for InstructPix2Pix training.
Implements targeted losses for better prediction of sparse elements like switches and routing points.
"""

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from .image_processing import (
    extract_color_pixels, 
    decode_latents_to_opencv_images,
    is_image_black
)

import cv2 

logger = get_logger(__name__, log_level="INFO")


def dice_loss(pred, target, mask, smooth=1e-6):
    """
    Compute Dice loss for masked regions.
    
    Args:
        pred: Predicted tensor [B, 3, H, W] in range [-1, 1]
        target: Target tensor [B, 3, H, W] in range [-1, 1]
        mask: Binary mask [B, 1, H, W] with 1 where to compute loss
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss scalar
    """
    # Apply mask to both pred and target
    pred_masked = pred * mask
    target_masked = target * mask
    
    # Flatten tensors
    pred_flat = pred_masked.view(pred.size(0), -1)
    target_flat = target_masked.view(target.size(0), -1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = (pred_flat * pred_flat).sum(dim=1)
    target_sum = (target_flat * target_flat).sum(dim=1)
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Return 1 - dice as loss (want to maximize dice, minimize loss)
    return (1.0 - dice).mean()


# def masked_l1_loss(pred, target, mask): # not used right now can be used instead of dice to compare approaches?
#     """
#     Compute L1 loss only in masked regions.
    
#     Args:
#         pred: Predicted tensor [B, 3, H, W]
#         target: Target tensor [B, 3, H, W] 
#         mask: Binary mask [B, 1, H, W]
    
#     Returns:
#         Masked L1 loss scalar
#     """
#     # Apply mask
#     pred_masked = pred * mask
#     target_masked = target * mask
    
#     # Compute L1 loss only where mask is active
#     loss = F.l1_loss(pred_masked, target_masked, reduction='none')
    
#     # Average over spatial dimensions and batch, but only where mask is active
#     mask_sum = mask.sum()
#     if mask_sum > 0:
#         return (loss * mask).sum() / mask_sum
#     else:
#         return torch.tensor(0.0, device=pred.device, requires_grad=True)

# def compute_auxiliary_losses(pred_latents, target_latents, vae, lambda_switch=5.0, lambda_routing=5.0, chunk_size=8):
#     """
#     Computes Dice-based auxiliary losses focused on green switches and purple routing points.
#     Decodes latents in chunks to avoid OOM issues.

#     Args:
#         pred_latents: Predicted latents [B, C, H, W]
#         target_latents: Target latents [B, C, H, W]
#         vae: VAE model (frozen or temporarily unfrozen)
#         chunk_size: Number of images to decode at once to avoid OOM
#         lambda_switch: Weight for the green switch Dice loss
#         lambda_routing: Weight for the purple routing Dice loss

#     Returns:
#         Dictionary with loss components and mask statistics.
#     """
#     import cv2  # make sure it's imported
#     device = pred_latents.device

#     # Decode in chunks to avoid OOM
#     decoded_preds, decoded_targets = [], []
#     for i in range(0, pred_latents.size(0), chunk_size):
#         pred_chunk = pred_latents[i:i+chunk_size]
#         target_chunk = target_latents[i:i+chunk_size]
#         with torch.no_grad():
#             decoded_pred = vae.decode(pred_chunk / vae.config.scaling_factor, return_dict=False)[0]
#             decoded_target = vae.decode(target_chunk / vae.config.scaling_factor, return_dict=False)[0]
#         decoded_preds.append(decoded_pred)
#         decoded_targets.append(decoded_target)

#     pred_images = torch.cat(decoded_preds, dim=0)
#     target_images = torch.cat(decoded_targets, dim=0)
#     B, _, H_dec, W_dec = pred_images.shape  # Use decoded image size, not latent shape

#     # Skip auxiliary loss if prediction is completely black
#     if all(is_image_black(pred_images[i]) for i in range(pred_images.size(0))):
#         logger.info("Skipping auxiliary loss: decoded image is black.")
#         return {
#             "loss_auxiliary": torch.tensor(0.0, device=device),
#             "loss_switch": torch.tensor(0.0, device=device),
#             "loss_routing": torch.tensor(0.0, device=device),
#             "switch_pixels": 0,
#             "routing_pixels": 0
#         }

#     # Decode latents to BGR OpenCV images for color mask extraction (resize to decoded image size)
#     pred_bgr_images = decode_latents_to_opencv_images(pred_latents, vae, chunk_size=chunk_size, resize=(W_dec, H_dec))

#     switch_masks, routing_masks = [], []

#     for bgr_img in pred_bgr_images:
#         # Switch mask (green hue: 30–90)
#         switch_mask = extract_color_pixels(bgr_img, lower_hue=30, upper_hue=90)
#         switch_mask = cv2.resize(switch_mask, (W_dec, H_dec), interpolation=cv2.INTER_NEAREST)
#         switch_tensor = torch.tensor(switch_mask / 255.0, device=device).unsqueeze(0).unsqueeze(0).expand(1, 3, H_dec, W_dec)
#         switch_masks.append(switch_tensor)

#         # Routing mask (purple hue: 130–150)
#         route_mask = extract_color_pixels(bgr_img, lower_hue=130, upper_hue=150)
#         route_mask = cv2.resize(route_mask, (W_dec, H_dec), interpolation=cv2.INTER_NEAREST)
#         route_tensor = torch.tensor(route_mask / 255.0, device=device).unsqueeze(0).unsqueeze(0).expand(1, 3, H_dec, W_dec)
#         routing_masks.append(route_tensor)

#     switch_masks = torch.cat(switch_masks, dim=0)
#     routing_masks = torch.cat(routing_masks, dim=0)

#     switch_pixel_count = switch_masks.sum().item()
#     routing_pixel_count = routing_masks.sum().item()

#     loss_switch = lambda_switch * dice_loss(pred_images, target_images, switch_masks) if switch_pixel_count > 0 else torch.tensor(0.0, device=device)
#     loss_routing = lambda_routing * dice_loss(pred_images, target_images, routing_masks) if routing_pixel_count > 0 else torch.tensor(0.0, device=device)

#     return {
#         "loss_auxiliary": loss_switch + loss_routing,
#         "loss_switch": loss_switch,
#         "loss_routing": loss_routing,
#         "switch_pixels": switch_pixel_count,
#         "routing_pixels": routing_pixel_count
#     }

def rgb_to_hsv_torch(image):
    """
    Convert a batch of RGB images to HSV in PyTorch.
    image: Tensor of shape [B, 3, H, W], values in [0, 1]
    Returns: HSV tensor of same shape
    """
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    maxc = torch.max(image, dim=1)[0]
    minc = torch.min(image, dim=1)[0]
    v = maxc

    s = torch.where(v == 0, torch.zeros_like(v), (maxc - minc) / (v + 1e-6))

    rc = (maxc - r) / (maxc - minc + 1e-6)
    gc = (maxc - g) / (maxc - minc + 1e-6)
    bc = (maxc - b) / (maxc - minc + 1e-6)

    h = torch.zeros_like(v)
    h[r == maxc] = (bc - gc)[r == maxc]
    h[g == maxc] = 2.0 + (rc - bc)[g == maxc]
    h[b == maxc] = 4.0 + (gc - rc)[b == maxc]
    h = (h / 6.0) % 1.0

    hsv = torch.stack([h, s, v], dim=1)
    return hsv

def hue_mask(hsv, hue_range, sharpness=100.0):
    """
    Create a soft hue mask using sigmoid approximation.
    hue_range: (low, high) in [0,1]
    Returns: mask of shape [B, 1, H, W]
    """
    h = hsv[:, 0:1]  # [B,1,H,W]
    low, high = hue_range
    low = torch.tensor(low, device=h.device)
    high = torch.tensor(high, device=h.device)
    mask = torch.sigmoid(sharpness * (h - low)) * (1 - torch.sigmoid(sharpness * (h - high)))
    return mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

def dice_loss(pred, target, mask, smooth=1e-6):
    assert pred.min() >= 0 and pred.max() <= 1, "Predictions out of bounds"
    assert mask.min() >= 0 and mask.max() <= 1, "Mask values not in [0, 1]"

    pred_masked = pred * mask
    target_masked = target * mask
    pred_flat = pred_masked.view(pred.size(0), -1)
    target_flat = target_masked.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = (pred_flat * pred_flat).sum(dim=1)
    target_sum = (target_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return (1.0 - dice).mean()

def compute_auxiliary_losses(pred_latents, target_latents, vae, lambda_switch=5.0, lambda_routing=5.0, chunk_size=8):
    device = pred_latents.device

    vae_dtype = next(vae.parameters()).dtype
    # Decode in chunks
    decoded_preds, decoded_targets = [], []
    for i in range(0, pred_latents.size(0), chunk_size):
        pred_chunk = pred_latents[i:i+chunk_size]
        target_chunk = target_latents[i:i+chunk_size]

        pred_chunk = pred_chunk.to(vae_dtype)
        target_chunk = target_chunk.to(vae_dtype)

        with torch.no_grad():
            decoded_pred = vae.decode(pred_chunk / vae.config.scaling_factor, return_dict=False)[0]
            decoded_target = vae.decode(target_chunk / vae.config.scaling_factor, return_dict=False)[0]

        decoded_preds.append(decoded_pred)
        decoded_targets.append(decoded_target)

        del decoded_pred, decoded_target
        #torch.cuda.empty_cache()

    pred_images = torch.cat(decoded_preds, dim=0).float()  # [B,3,H,W]
    target_images = torch.cat(decoded_targets, dim=0).float() 

    pred_norm = (pred_images.clamp(-1, 1) + 1) / 2  # [0,1] for HSV
    target_norm = (target_images.clamp(-1, 1) + 1) / 2

    hsv = rgb_to_hsv_torch(pred_norm)

    # Define hue ranges in [0,1]
    switch_mask = hue_mask(hsv, (30/360, 90/360)).float() 
    routing_mask = hue_mask(hsv, (130/360, 150/360)).float() 

    loss_switch  = lambda_switch  * dice_loss(pred_norm, target_norm, switch_mask)  if switch_mask.sum()>0 else torch.tensor(0.0, device=device)
    loss_routing = lambda_routing * dice_loss(pred_norm, target_norm, routing_mask) if routing_mask.sum()>0 else torch.tensor(0.0, device=device)
    loss_aux     = loss_switch + loss_routing

    # 3) stash only the scalars you need
    out = {
        "loss_auxiliary": loss_aux,
        "loss_switch":    loss_switch,
        "loss_routing":   loss_routing,
        "switch_pixels":  switch_mask.sum().item(),
        "routing_pixels": routing_mask.sum().item(),
    }

    # 4) now delete the big tensors & lists
    del decoded_preds, decoded_targets
    del pred_images, target_images, pred_norm, target_norm, hsv
    del switch_mask, routing_mask
    import gc; gc.collect()

    # 5) return just the small dict
    return out