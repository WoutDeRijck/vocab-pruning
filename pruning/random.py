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

def random_based_pruning(token_counter, prune_percent, min_tokens=5, random_seed=42, 
                        param_based=False, embedding_dim=768, total_params=None):
    """
    Prune vocabulary based on random selection.
    This serves as a baseline approach.
    
    Args:
        token_counter: Counter with token frequencies
        prune_percent: Percentage of tokens or parameters to prune
        min_tokens: Minimum number of tokens to keep
        random_seed: Random seed for reproducibility
        param_based: If True, prune based on parameter percentage rather than token percentage
        embedding_dim: Dimension of token embeddings (needed for parameter calculation)
        total_params: Total parameters in the model (needed for parameter-based pruning)
        
    Returns:
        tokens_to_keep: List of token IDs to keep
        tokens_to_remove: List of token IDs to remove
    """
    if param_based:
        logger.info(f"Performing parameter-based random pruning with prune_percent={prune_percent}% of total parameters")
    else:
        logger.info(f"Performing token-based random pruning with prune_percent={prune_percent}% of tokens")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all token IDs from the dataset (these are the train tokens)
    train_token_ids = list(token_counter.keys())
    logger.info(f"Dataset contains {len(train_token_ids)} unique tokens")
    
    # Always keep special tokens (first 5 tokens are usually special tokens in ModernBERT)
    special_tokens = list(range(5))
    
    # Get non-special tokens from the train set
    train_non_special_tokens = [token_id for token_id in train_token_ids if token_id not in special_tokens]
    logger.info(f"Dataset contains {len(train_non_special_tokens)} non-special tokens")
    
    # Calculate number of tokens to keep based on pruning strategy
    if param_based and total_params is not None:
        # Calculate number of embedding parameters to remove
        params_to_remove = (prune_percent / 100) * total_params
        
        # Convert params to remove to tokens to remove
        tokens_to_remove_count = int(params_to_remove / embedding_dim)
        
        # We can only remove tokens from the training set, not more
        max_removable = len(train_non_special_tokens)
        tokens_to_remove_count = min(tokens_to_remove_count, max_removable)
        
        # Calculate tokens to keep from the training set
        num_tokens_to_keep = max(min_tokens, len(train_non_special_tokens) - tokens_to_remove_count)
        
        logger.info(f"Target parameter reduction: {params_to_remove:,} parameters")
        logger.info(f"This equals removing {tokens_to_remove_count} tokens at {embedding_dim} dimensions each")
        logger.info(f"Will keep {num_tokens_to_keep} non-special tokens from the training set")
    else:
        # Traditional token-based pruning
        num_prunable_tokens = len(train_non_special_tokens)
        num_tokens_to_keep = max(min_tokens, int(num_prunable_tokens * (1 - prune_percent / 100)))
    
    # Randomly select tokens to keep from the training set
    tokens_to_keep_non_special = random.sample(train_non_special_tokens, num_tokens_to_keep)
    
    # Combine special tokens and randomly selected tokens
    tokens_to_keep = special_tokens + tokens_to_keep_non_special
    
    # Get tokens to remove from the training set
    tokens_to_remove = [token_id for token_id in train_token_ids if token_id not in tokens_to_keep]
    
    # Calculate parameter reduction achieved
    param_reduction = len(tokens_to_remove) * embedding_dim
    param_reduction_percent = 100 * (param_reduction / (len(train_token_ids) * embedding_dim))
    
    logger.info(f"Kept {len(tokens_to_keep)} tokens, removed {len(tokens_to_remove)} tokens")
    logger.info(f"This reduces embedding parameters by {param_reduction:,} parameters ({param_reduction_percent:.2f}%)")
    
    if param_based and total_params is not None:
        overall_param_reduction_percent = 100 * (param_reduction / total_params)
        logger.info(f"Overall parameter reduction: {overall_param_reduction_percent:.2f}% of total model parameters")
    
    return tokens_to_keep, tokens_to_remove

def setup_random_based_model(task_name, model_name, prune_percent=0, random_seed=42, param_based=False):
    """
    Set up a model with reduced vocabulary based on random pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens or parameters to prune
        random_seed: Random seed for reproducibility
        param_based: If True, prune based on parameter percentage rather than token percentage
        
    Returns:
        model: Model with reduced vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: None for random-based method
    """
    pruning_type = "parameter-based" if param_based else "token-based"
    logger.info(f"Setting up {pruning_type} random model for {task_name} with {prune_percent}% pruning")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with counts - train-only tokens
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, all_token_ids = get_dataset_tokens_with_counts(vocab_name, train_only=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Get total model parameters and embedding dimension
    total_params = sum(p.numel() for p in model.parameters())
    embedding_dim = model.model.embeddings.tok_embeddings.embedding_dim
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Token embeddings: {model.model.embeddings.tok_embeddings.num_embeddings:,} tokens with dimension {embedding_dim}")
    
    # Apply random-based pruning if requested
    if prune_percent > 0:
        tokens_to_keep, _ = random_based_pruning(
            token_counter, 
            prune_percent, 
            random_seed=random_seed,
            param_based=param_based,
            embedding_dim=embedding_dim,
            total_params=total_params
        )
    else:
        tokens_to_keep = sorted(list(all_token_ids))
    
    # Create reduced embeddings
    logger.info(f"Creating reduced embeddings for task_vocab of size {len(tokens_to_keep)}")
    token_map, reduced_embeddings = create_reduced_embeddings(tokens_to_keep, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, None  # No OOV lookup for random method 