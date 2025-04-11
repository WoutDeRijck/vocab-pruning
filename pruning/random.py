"""
Random Selection vocabulary pruning.

This module implements vocabulary pruning based on random selection.
This serves as a baseline approach where tokens are pruned randomly
without consideration for importance.
"""

import logging
import random
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .base import get_dataset_tokens_with_counts, create_reduced_embeddings
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def random_based_pruning(token_counter, prune_percent, min_tokens=5, random_seed=42):
    """
    Prune vocabulary based on random selection.
    This serves as a baseline approach.
    
    Args:
        token_counter: Counter with token frequencies
        prune_percent: Percentage of tokens to prune
        min_tokens: Minimum number of tokens to keep
        random_seed: Random seed for reproducibility
        
    Returns:
        tokens_to_keep: List of token IDs to keep
        tokens_to_remove: List of token IDs to remove
    """
    logger.info(f"Performing random-based pruning with prune_percent={prune_percent}%")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all token IDs
    all_token_ids = list(token_counter.keys())
    
    # Always keep special tokens (first 5 tokens are usually special tokens in ModernBERT)
    special_tokens = list(range(5))
    
    # Calculate number of tokens to keep (excluding special tokens)
    num_prunable_tokens = len(all_token_ids) - len(special_tokens)
    num_tokens_to_keep = max(min_tokens, int(num_prunable_tokens * (1 - prune_percent / 100)))
    
    # Get non-special tokens
    non_special_tokens = [token_id for token_id in all_token_ids if token_id not in special_tokens]
    
    # Randomly select tokens to keep
    tokens_to_keep_non_special = random.sample(non_special_tokens, num_tokens_to_keep)
    
    # Combine special tokens and randomly selected tokens
    tokens_to_keep = special_tokens + tokens_to_keep_non_special
    
    # Get tokens to remove
    tokens_to_remove = [token_id for token_id in all_token_ids if token_id not in tokens_to_keep]
    
    logger.info(f"Kept {len(tokens_to_keep)} tokens, removed {len(tokens_to_remove)} tokens")
    
    return tokens_to_keep, tokens_to_remove

def setup_random_based_model(task_name, model_name, prune_percent=0, random_seed=42):
    """
    Set up a model with reduced vocabulary based on random pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune randomly
        random_seed: Random seed for reproducibility
        
    Returns:
        model: Model with reduced vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: None for random-based method
    """
    logger.info(f"Setting up random-based model for {task_name} with {prune_percent}% pruning")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with counts
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, all_token_ids = get_dataset_tokens_with_counts(vocab_name, train_only=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Apply random-based pruning if requested
    if prune_percent > 0:
        tokens_to_keep, _ = random_based_pruning(token_counter, prune_percent, random_seed=random_seed)
    else:
        tokens_to_keep = sorted(list(all_token_ids))
    
    # Create reduced embeddings
    token_map, reduced_embeddings = create_reduced_embeddings(tokens_to_keep, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, None  # No OOV lookup for random method 