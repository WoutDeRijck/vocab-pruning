"""
Data handling for vocabulary pruning.

This module contains functions for dataset preparation and data collators.
"""

import logging
import torch
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset

# Configure logging
logger = logging.getLogger(__name__)

class ReducedVocabDataCollator:
    """
    Data collator for reduced vocabulary models.
    Used by clustering-based pruning.
    Handles padding and preparing batches for the model.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, features):
        # Get max length in batch
        max_length = max(len(x['input_ids']) for x in features)
        
        # Pad input_ids and attention masks
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            # Pad input_ids
            padded = f['input_ids'] + [self.pad_token_id] * (max_length - len(f['input_ids']))
            input_ids.append(padded)
            
            # Pad attention mask
            mask = f['attention_mask'] + [0] * (max_length - len(f['attention_mask']))
            attention_mask.append(mask)
            
            # Get labels
            labels.append(f['labels'])
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

class HybridCollator:
    """
    Data collator for hybrid vocabulary models.
    Used by hybrid and importance-based pruning.
    Handles padding and remapping OOV tokens to appropriate clusters.
    """
    def __init__(self, pad_token_id, unk_token_id, oov_lookup=None):
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.oov_lookup = oov_lookup or {}
        
    def __call__(self, features):
        # Get max length in batch
        max_length = max(len(x['input_ids']) for x in features)
        
        # Pad input_ids and attention masks
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            # Pad input_ids
            padded = f['input_ids'] + [self.pad_token_id] * (max_length - len(f['input_ids']))
            input_ids.append(padded)
            
            # Pad attention mask
            mask = f['attention_mask'] + [0] * (max_length - len(f['attention_mask']))
            attention_mask.append(mask)
            
            # Get labels
            labels.append(f['labels'])
        
        # Check if any labels are -1 (test set)
        has_invalid_labels = any(label == -1 for label in labels)
        
        result = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
        
        # If we have invalid labels, add a flag to indicate prediction-only mode
        if has_invalid_labels:
            result['prediction_only'] = True
            
        return result

def prepare_datasets_with_mapping(task_name, token_map, oov_lookup=None, tokenizer=None, batch_size=512):
    """
    Prepare datasets for training with reduced vocabulary with GPU acceleration.
    Works for all pruning methods.
    
    Args:
        task_name: Name of the GLUE task
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping from OOV token ID to cluster representative ID (for hybrid methods)
        tokenizer: Tokenizer for the original model
        batch_size: Batch size for preprocessing
        
    Returns:
        train_dataset: Training dataset with remapped token IDs
        eval_dataset: Evaluation dataset with remapped token IDs
        test_dataset: Test dataset with remapped token IDs (if available)
    """
    # Import task metadata
    from utils import get_task_metadata, task_to_keys
    
    # Set multiprocessing method for tokenizer
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    
    # Load dataset
    dataset_name = "mnli" if task_name.startswith("mnli") else task_name
    raw_datasets = load_dataset("glue", dataset_name)
    
    # Handle 'mnli' task name specially for task metadata
    task_meta = get_task_metadata(task_name)
    train_ds_name = task_meta["dataset_names"]["train"]
    valid_ds_name = task_meta["dataset_names"]["valid"]

    # Print available splits for debugging
    logger.info(f"Available splits: {list(raw_datasets.keys())}")
    logger.info(f"Using train: {train_ds_name}, validation: {valid_ds_name}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get the UNK token ID in the new mapping
    unk_token_id = 0
    
    # Create tensor mapping for both direct mapping and OOV lookup
    # Create a tensor mapping from old to new token IDs
    # Add +1 to max_token_id to handle the case where the max token ID is exactly at the boundary
    max_token_id = max(tokenizer.vocab_size, max(token_map.keys()) + 2)  # Add extra padding to avoid index errors
    id_map_tensor = torch.full((max_token_id,), unk_token_id, dtype=torch.long, device=device)
    
    # First fill with direct mappings
    for old_id, new_id in token_map.items():
        id_map_tensor[old_id] = new_id
    
    # Then add OOV lookup mappings if provided
    if oov_lookup:
        for old_id, new_id in oov_lookup.items():
            id_map_tensor[old_id] = new_id
    
    sentence1_key, sentence2_key = task_to_keys[dataset_name]
    
    # Process a split using efficient GPU batching
    def transform_split_gpu(dataset_split):
        # For large datasets, use larger batch sizes to maximize GPU utilization
        if torch.cuda.is_available():
            # Calculate batch size based on GPU memory
            mem_info = torch.cuda.mem_get_info()
            free_mem = mem_info[0] / 1024**2  # Free memory in MB
            total_mem = mem_info[1] / 1024**2  # Total memory in MB
            
            # Adjust batch size based on GPU memory
            if "mnli" in task_name:
                if total_mem > 40000:  # For GPUs with more than 40GB VRAM
                    process_batch_size = 4096
                elif total_mem > 24000:  # For GPUs with 24-40GB VRAM
                    process_batch_size = 2048
                else:  # For smaller GPUs
                    process_batch_size = 1024
            else:
                # For smaller datasets, we can use more aggressive settings
                if total_mem > 40000:  # For large GPUs
                    process_batch_size = 8192
                elif total_mem > 24000:  # For medium GPUs
                    process_batch_size = 4096
                else:  # For smaller GPUs
                    process_batch_size = 2048
            
            logger.info(f"Using batch size {process_batch_size} for dataset processing based on GPU memory: {total_mem:.2f}MB")
        else:
            # CPU fallback
            process_batch_size = 256
        
        all_items = []
        
        # Process in batches
        for i in tqdm(range(0, len(dataset_split[sentence1_key]), process_batch_size),
                     desc="Processing examples"):
            # Get batch texts
            batch_texts1 = dataset_split[sentence1_key][i:i+process_batch_size]
            batch_texts2 = None if sentence2_key is None else dataset_split[sentence2_key][i:i+process_batch_size]
            
            # Try to get labels, but handle cases where labels might be missing or -1
            try:
                batch_labels = dataset_split['label'][i:i+process_batch_size]
            except (KeyError, IndexError):
                # If labels are not available, use dummy values (0) 
                # These won't be used for evaluation on test sets
                batch_labels = [0] * len(batch_texts1)
            
            # Tokenize with optimal settings
            if batch_texts2 is None:
                encodings = tokenizer(
                    batch_texts1, 
                    padding='max_length', 
                    truncation=True,
                    max_length=128,  # Shorter max_length for faster processing
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
            else:
                encodings = tokenizer(
                    batch_texts1, 
                    batch_texts2, 
                    padding='max_length', 
                    truncation=True,
                    max_length=128,  # Shorter max_length for faster processing
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
            
            # Move to GPU for faster processing
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            
            # Ensure no out-of-bounds indices
            input_ids = torch.clamp(input_ids, 0, max_token_id - 1)
            
            # Map token IDs using the mapping tensor (on GPU)
            new_input_ids = id_map_tensor[input_ids]
            
            # Create batch items
            for j in range(len(new_input_ids)):
                item = {
                    'input_ids': new_input_ids[j].cpu().tolist(),
                    'attention_mask': attention_mask[j].cpu().tolist(),
                    'labels': batch_labels[j]
                }
                all_items.append(item)
            
            # Clear GPU cache every few batches
            if i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return Dataset.from_list(all_items)
    
    # Transform train and validation sets
    logger.info(f"Processing train split ({len(raw_datasets[train_ds_name])} examples)...")
    train_dataset = transform_split_gpu(raw_datasets[train_ds_name])
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"Processing validation split ({len(raw_datasets[valid_ds_name])} examples)...")
    eval_dataset = transform_split_gpu(raw_datasets[valid_ds_name])
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get test dataset name if available
    test_ds_name = task_meta["dataset_names"].get("test")
    
    # Process test set if available
    if test_ds_name and test_ds_name in raw_datasets:
        logger.info(f"Processing test split ({len(raw_datasets[test_ds_name])} examples)...")
        test_dataset = transform_split_gpu(raw_datasets[test_ds_name])
    else:
        test_dataset = None
    
    return train_dataset, eval_dataset, test_dataset 