"""
Image processing utilities for InstructPix2Pix training.
Contains functions for color detection, region extraction, and image conversions.
"""

import cv2
import numpy as np
import PIL
import requests
import torch
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT


def convert_to_np(image, resolution):
    """Convert PIL image to numpy array with specified resolution."""
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    """Download image from URL or load from local path."""
    if url.startswith("http://") or url.startswith("https://"):
        image = PIL.Image.open(requests.get(url, stream=True, timeout=DIFFUSERS_REQUEST_TIMEOUT).raw)
    else:
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


def extract_color_pixels(image: np.ndarray, lower_hue: int = 30, upper_hue: int = 90, 
                        saturation_threshold: int = 30, value_threshold: int = 20) -> np.ndarray:
    """
    Extract colored pixels from the image with a given tolerance.
    
    Parameters:
    - image: Input image in BGR format.
    - lower_hue: Lower bound for the hue value for color to be extracted
    - upper_hue: Upper bound for the hue value for color to be extracted
    - saturation_threshold: Minimum saturation value to consider.
    - value_threshold: Minimum brightness value to consider.

    Returns:
    - color_mask: Mask of the same size as the image, with white pixels representing colored areas.
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

def match_points(predicted, gt):
    """Match predicted points to ground truth points using nearest neighbor."""

    n_pred_orig = len(predicted)
    n_gt       = len(gt)


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
    
    matching_score = (n_pred_orig - n_gt) / n_gt if n_gt else 0  # 0% best positive too many predictions, negative too few predictions
    # used to be matching_score = (len(predicted) - len(gt)) / len(gt) if gt else 0 -> if # predictions <= # in gt, then will be -1, else will be 

    return matched_predicted, unmatched_predicted, avg_score, matching_score

def decode_latents_to_opencv_images(latents, vae, chunk_size=8, resize=None):
    """
    Decodes a batch of latents to OpenCV BGR images, using smaller chunks to avoid OOM.
    
    Args:
        latents: Tensor [B, C, H, W]
        vae: VAE model
        chunk_size: Chunk size for decoding to prevent OOM
        resize: Tuple (H, W) to resize output images (e.g., (256, 256)). If None, no resizing.
    
    Returns:
        List of BGR uint8 images
    """
    all_images = []

    for i in range(0, latents.shape[0], chunk_size):
        with torch.no_grad():
            latents_chunk = latents[i:i+chunk_size]
            decoded = vae.decode(latents_chunk / vae.config.scaling_factor, return_dict=False)[0]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
            decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()  # BCHW -> BHWC

            for img in decoded:
                img = (img * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if resize is not None:
                    img_bgr = cv2.resize(img_bgr, resize, interpolation=cv2.INTER_AREA)
                all_images.append(img_bgr)

    return all_images



def is_image_black(image_tensor_or_pil, threshold=0):
    """
    Check if an image is completely black (or nearly black).
    
    Args:
        image_tensor_or_pil: PIL Image or tensor in any format
        threshold: Minimum brightness threshold (0-255 for PIL, -1 to 1 for tensor)
        
    Returns:
        bool: True if image is considered black
    """
    if hasattr(image_tensor_or_pil, 'getextrema'):  # PIL Image
        extrema = image_tensor_or_pil.getextrema()
        return all(channel_ext[1] <= threshold for channel_ext in extrema)
    else:  # Tensor
        # Convert tensor range if needed
        if image_tensor_or_pil.min() >= -1 and image_tensor_or_pil.max() <= 1:
            # Tensor in [-1, 1] range
            threshold_tensor = (threshold / 127.5) - 1  # Convert to [-1, 1] range
        else:
            # Assume tensor in [0, 255] range
            threshold_tensor = threshold
            
        return image_tensor_or_pil.max() <= threshold_tensor
