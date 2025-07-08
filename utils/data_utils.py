"""
Data processing utilities for InstructPix2Pix training.
Contains functions for dataset loading, preprocessing, and collate functions.
"""

import torch
import logging
from torchvision import transforms

logger = logging.getLogger(__name__)

def create_dataloaders(args, train_dataset, val_dataset, tokenizer, multiprocessing):
    """
    Create optimized data loaders for training and validation.
    
    Args:
        args: Training arguments
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer for processing text prompts
        multiprocessing: Multiprocessing module for CPU count
        
    Returns:
        train_dataloader, val_dataloader: DataLoader objects for training and validation
    """
    # Collate function for just-in-time image decoding and processing
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
                transform(example[args.original_image_column]) for example in batch
            ])
            original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()

            edited_pixel_values = torch.stack([
                transform(example[args.edited_image_column]) for example in batch
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

    # Configure worker count based on system resources
    max_workers = min(args.dataloader_num_workers, multiprocessing.cpu_count())
    
    # Use conservative batch sizes for validation to avoid CUDA OOM
    # Validation uses significantly more memory due to:
    # 1. Classifier-free guidance (2x embeddings)
    # 2. Image guidance (additional image embeddings)
    # 3. Multiple denoising steps with concatenated latents
    val_batch_size = 32

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
        num_workers=max(1, max_workers // 2),  # Use fewer workers for validation
        pin_memory=True,
        persistent_workers=False,
    )
    
    logger.info(f"Created dataloaders with {max_workers} workers:")
    logger.info(f"  - Training: batch_size={args.train_batch_size}, {len(train_dataset)} samples")
    logger.info(f"  - Validation: batch_size={val_batch_size}, {len(val_dataset)} samples")
    
    return train_dataloader, val_dataloader


def preprocess_dataset(dataset, args, tokenizer):
    """
    Preprocess the dataset for training and validation.
    
    Args:
        dataset: The raw dataset
        args: Training arguments
        tokenizer: Tokenizer for processing text prompts
        
    Returns:
        train_dataset, val_dataset: Preprocessed datasets
    """
    # Tokenize captions helper function
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return inputs.input_ids
    
    # Extract column names from arguments or defaults
    edit_prompt_column = args.edit_prompt_column
    
    # Simple preprocessing that only handles tokenization - images processed on-the-fly
    def preprocess_fn(example):
        # Tokenize and convert to list (HuggingFace datasets handle lists better than tensors)
        input_ids = tokenize_captions([example[edit_prompt_column]])
        example["input_ids"] = input_ids.squeeze(0).tolist()  # Convert to list for HF compatibility
        return example

    # Apply minimal preprocessing (just tokenization)
    train_dataset = dataset["train"].map(preprocess_fn, remove_columns=[edit_prompt_column])
    val_dataset = dataset["validation"].map(preprocess_fn, remove_columns=[edit_prompt_column])
    
    logger.info(f"Preprocessed datasets: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_dataset, val_dataset
