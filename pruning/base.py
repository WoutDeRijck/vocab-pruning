"""
Base utilities for vocabulary pruning.

This module contains common functions and utilities used by all pruning methods.
"""

import logging
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# GLUE task input keys mapping
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Get the basic dataset vocabulary (for clustering-based pruning)
def get_dataset_vocabulary(task_name, train_only=True):
    """
    Extract basic vocabulary from a GLUE task dataset.
    Used for clustering-based pruning.
    
    Args:
        task_name: Name of the GLUE task
        train_only: Whether to use only the training set
        
    Returns:
        List of token IDs forming the task-specific vocabulary
    """
    logger.info(f"Getting dataset vocabulary for {task_name} with train_only={train_only}")
    
    # Load the dataset
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Set to store unique token IDs
    unique_token_ids = set()
    
    # Determine which splits to process
    splits_to_process = ['train'] if train_only else raw_datasets.keys()
    
    # Process selected splits
    for split in splits_to_process:
        if split not in raw_datasets:
            logger.warning(f"Split {split} not found in dataset, skipping.")
            continue
            
        # Process sentence1
        texts = raw_datasets[split][sentence1_key]
        for text in texts:
            # Get token IDs instead of tokens
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            unique_token_ids.update(token_ids)
        
        # Process sentence2 if it exists
        if sentence2_key is not None:
            texts = raw_datasets[split][sentence2_key]
            for text in texts:
                if text is not None:
                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                    unique_token_ids.update(token_ids)
    
    # Convert to sorted list for consistent mapping
    task_vocab = sorted(list(unique_token_ids))
    logger.info(f"Found {len(task_vocab)} unique tokens in the dataset")
    
    return task_vocab

# Get vocabulary with frequency counts (for frequency-based and hybrid pruning)
def get_dataset_tokens_with_counts(task_name, train_only=True):
    """
    Extract vocabulary from a GLUE task dataset with token counts.
    Used for frequency-based and hybrid pruning.
    
    Args:
        task_name: Name of the GLUE task
        train_only: Whether to use only the training set
        
    Returns:
        token_counter: Counter with token_id -> count mappings
        all_token_ids: Set of all token IDs seen in the dataset
    """
    logger.info(f"Getting dataset vocabulary with counts for {task_name}, train_only={train_only}")
    
    # Load the dataset
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Counter to store token counts
    token_counter = Counter()
    
    # Set to store all unique token IDs (including low frequency)
    all_token_ids = set()
    
    # Determine which splits to process
    splits_to_process = ['train'] if train_only else raw_datasets.keys()
    
    # Process selected splits
    for split in splits_to_process:
        if split not in raw_datasets:
            logger.warning(f"Split {split} not found in dataset, skipping.")
            continue
            
        # Process sentence1
        texts = raw_datasets[split][sentence1_key]
        for text in texts:
            # Get token IDs instead of tokens
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            token_counter.update(token_ids)
            all_token_ids.update(token_ids)
        
        # Process sentence2 if it exists
        if sentence2_key is not None:
            texts = raw_datasets[split][sentence2_key]
            for text in texts:
                if text is not None:
                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                    token_counter.update(token_ids)
                    all_token_ids.update(token_ids)
    
    logger.info(f"Found {len(all_token_ids)} unique tokens in the dataset")
    
    return token_counter, all_token_ids

# Create reduced embeddings (used by clustering and frequency pruning)
def create_reduced_embeddings(task_vocab, model):
    """
    Create a reduced embedding matrix based on task-specific vocabulary.
    Used by clustering-based pruning.
    
    Args:
        task_vocab: List of token IDs to include in reduced vocabulary
        model: Model whose embeddings will be reduced
        
    Returns:
        token_map: Mapping from original token IDs to new consecutive IDs
        reduced_embeddings: Reduced embedding matrix containing only needed vectors
    """
    logger.info(f"Creating reduced embeddings for task_vocab of size {len(task_vocab)}")
    
    # Create mapping from original token IDs to new consecutive IDs
    token_map = {old_id: new_id for new_id, old_id in enumerate(task_vocab)}
    
    # Get original embedding matrix
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    
    # Create new embedding matrix with only the needed vectors
    reduced_embeddings = torch.stack([original_embeddings[i] for i in task_vocab])
    
    return token_map, reduced_embeddings

# Create hybrid embeddings (used by hybrid and importance pruning)
def create_hybrid_embeddings(tokens_to_keep, oov_token_map, cluster_centers, model):
    """
    Create reduced embedding matrix with hybrid token mapping.
    Used by hybrid and importance-based pruning.
    
    Args:
        tokens_to_keep: List of token IDs to keep
        oov_token_map: Mapping from removed token ID to cluster center token ID
        cluster_centers: List of token IDs for cluster centers
        model: Model whose embeddings will be used
        
    Returns:
        token_map: Mapping from original token IDs to new consecutive IDs
        reduced_embeddings: Reduced embedding matrix
        oov_lookup: Lookup table for mapping OOV tokens
    """
    logger.info(f"Creating hybrid embeddings with {len(tokens_to_keep)} kept tokens and {len(cluster_centers)} OOV clusters")
    
    # Get original embedding matrix
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    
    # Combine kept tokens and cluster centers (avoiding duplicates)
    all_tokens_to_keep = list(tokens_to_keep)
    for center in cluster_centers:
        if center not in all_tokens_to_keep:
            all_tokens_to_keep.append(center)
    
    # Create mapping from original token IDs to new consecutive IDs
    token_map = {old_id: new_id for new_id, old_id in enumerate(all_tokens_to_keep)}
    
    # Create reduced embedding matrix with only the needed vectors
    reduced_embeddings = torch.stack([original_embeddings[i] for i in all_tokens_to_keep])
    
    # Create OOV lookup that maps original token IDs to their cluster representatives
    oov_lookup = {}
    for orig_id, center_id in oov_token_map.items():
        if center_id in token_map:  # Ensure the center ID is in our vocabulary
            oov_lookup[orig_id] = token_map[center_id]
    
    return token_map, reduced_embeddings, oov_lookup 