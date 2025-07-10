import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionInstructPix2PixPipeline
from utils.auxiliary_loss import compute_auxiliary_losses
from utils.image_processing import decode_latents_to_opencv_images, extract_color_pixels
from torchvision import transforms
from itertools import islice

# ----------- Config -----------
BATCH_SIZE = 64
SAVE_DIR = "debug_aux_loss"
MODEL_DIR = "flexgen_diffusion"
DATASET_NAME = "lukalafaye/NoC_with_dots"
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# Load pipeline
print("🔧 Loading fine-tuned pipeline...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    safety_checker=None,
    revision="fp16"
).to(device)
vae = pipe.vae.eval()

# Load 64 examples from streaming dataset
print("📦 Loading dataset...")
dataset = list(islice(load_dataset(DATASET_NAME, split="train", streaming=True), BATCH_SIZE))

# Define transform to match training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

print("🖼️ Preprocessing images...")
pixel_values = torch.stack([
    transform(example["edited_image"]) for example in dataset
]).to(device, dtype=torch.float16)

# Encode latents
print("🎯 Encoding target latents...")
with torch.no_grad():
    target_latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

# Simulate predicted latents with noise
pred_latents = target_latents + 0.1 * torch.randn_like(target_latents)

# Compute auxiliary losses
print("📊 Computing auxiliary losses...")
with torch.no_grad():
    individual_losses = []
    for i in range(BATCH_SIZE):
        pred_i = pred_latents[i:i+1]          # shape [1, C, H, W]
        target_i = target_latents[i:i+1]      # shape [1, C, H, W]
        loss_i = compute_auxiliary_losses(pred_i, target_i, vae, chunk_size=1)
        individual_losses.append(loss_i)

# Decode predicted and GT latents to OpenCV BGR images
print("🖼️ Decoding images...")
# pred_images = decode_latents_to_opencv_images(pred_latents, vae, chunk_size=8)

pred_images = decode_latents_to_opencv_images(pred_latents, vae, chunk_size=8, resize=(256, 256))
gt_images = decode_latents_to_opencv_images(target_latents, vae, chunk_size=8, resize=(256, 256))

# Save 64 image panels
print("💾 Saving visual debug outputs...")
for i in range(64):
    pred = pred_images[i]
    gt = gt_images[i]

    # Get masks
    switch_mask = extract_color_pixels(pred, lower_hue=30, upper_hue=90)
    routing_mask = extract_color_pixels(pred, lower_hue=130, upper_hue=150)

    # Make masks BGR
    switch_mask_rgb = cv2.cvtColor(switch_mask, cv2.COLOR_GRAY2BGR)
    routing_mask_rgb = cv2.cvtColor(routing_mask, cv2.COLOR_GRAY2BGR)

    # Combine all horizontally
    combined = np.hstack([pred, gt, switch_mask_rgb, routing_mask_rgb])

    # Overlay loss values
    assert individual_losses[i]["loss_switch"].item() >= 0
    assert individual_losses[i]["loss_routing"].item() >= 0

    cv2.putText(combined, f"Switch loss: {individual_losses[i]['loss_switch'].item():.4f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, f"Routing loss: {individual_losses[i]['loss_routing'].item():.4f}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2)


    # Save image
    cv2.imwrite(os.path.join(SAVE_DIR, f"sample_{i}.jpg"), combined)

# print("✅ Done. Saved 4 samples and computed auxiliary loss:")
# for k, v in loss_dict.items():
#     print(f"  {k}: {v}")



print("🔁 Testing VAE decode → encode cycle consistency...")

# Re-encode with .sample() instead of .mode()
with torch.no_grad():
    target_latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

# Decode in chunks
decoded_batches = []
with torch.no_grad():
    for i in range(0, target_latents.size(0), 8):
        latents_chunk = target_latents[i:i+8]
        decoded = vae.decode(latents_chunk / vae.config.scaling_factor, return_dict=False)[0]
        decoded_batches.append(decoded)

decoded_images_tensor = torch.cat(decoded_batches, dim=0)

# Re-encode the decoded images (using .sample again)
with torch.no_grad():
    reencoded_latents = vae.encode(decoded_images_tensor).latent_dist.sample() * vae.config.scaling_factor

# Compute loss
with torch.no_grad():
    reconstruction_loss = compute_auxiliary_losses(reencoded_latents, target_latents, vae)

print("🔍 Reconstruction loss from decode → encode:")
for k, v in reconstruction_loss.items():
    print(f"  {k}: {v:.6f}")




print("🔁 Testing VAE decode → pixel → compare in chunks...")

chunk_size = 8
recon_images = []
original_images = []

with torch.no_grad():
    for i in range(0, pixel_values.shape[0], chunk_size):
        # Get batch
        px_chunk = pixel_values[i:i+chunk_size]
        
        # Encode and decode
        latents = vae.encode(px_chunk).latent_dist.mode() * vae.config.scaling_factor
        recon_chunk = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

        # Unnormalize both
        recon_chunk = (recon_chunk.clamp(-1, 1) + 1) / 2  # [0,1]
        px_chunk = (px_chunk.clamp(-1, 1) + 1) / 2        # [0,1]

        recon_images.append(recon_chunk)
        original_images.append(px_chunk)

# Concatenate all chunks
recon_images = torch.cat(recon_images, dim=0)
original_images = torch.cat(original_images, dim=0)

# Compute reconstruction MSE in pixel space
mse = torch.nn.functional.mse_loss(recon_images, original_images).item()
print(f"🔍 VAE pixel-space reconstruction MSE: {mse:.6f}")
