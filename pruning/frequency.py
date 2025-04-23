"""
Frequency-based vocabulary pruning.

This module implements vocabulary pruning based on token frequency.
"""

import logging
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .base import get_dataset_tokens_with_counts, create_reduced_embeddings
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def frequency_based_pruning(token_counter, prune_percent, min_tokens=5, 
                          param_based=False, embedding_dim=768, total_params=None,
                          total_vocab_size=None):
    """
    Prune vocabulary based on token frequency.
    Used by frequency-based and hybrid pruning methods.
    
    Args:
        token_counter: Counter with token frequencies
        prune_percent: Percentage of tokens or parameters to prune
        min_tokens: Minimum number of tokens to keep
        param_based: If True, prune based on parameter percentage rather than token percentage
        embedding_dim: Dimension of token embeddings (needed for parameter calculation)
        total_params: Total parameters in the model (needed for parameter-based pruning)
        total_vocab_size: Total size of the original vocabulary
        
    Returns:
        tokens_to_keep: List of token IDs to keep
        tokens_to_remove: List of token IDs to remove
    """
    if param_based:
        logger.info(f"Performing parameter-based frequency pruning with prune_percent={prune_percent}% of total parameters")
    else:
        logger.info(f"Performing token-based frequency pruning with prune_percent={prune_percent}% of tokens")
    
    # Sort tokens by frequency (most common first)
    sorted_tokens = token_counter.most_common()
    train_token_ids = [token_id for token_id, _ in sorted_tokens]
    logger.info(f"Dataset contains {len(train_token_ids)} unique tokens")
    
    # Always keep special tokens (first 5 tokens are usually special tokens in ModernBERT)
    special_tokens = list(range(5))
    
    # Calculate parameter reduction achieved by keeping only training tokens
    if param_based and total_params is not None and total_vocab_size is not None:
        # Calculate parameter reduction from keeping only train tokens
        full_emb_params = total_vocab_size * embedding_dim
        train_emb_params = len(train_token_ids) * embedding_dim
        param_reduction_from_train_only = full_emb_params - train_emb_params
        
        # Calculate what percentage of total parameters this represents
        train_only_reduction_percent = 100 * (param_reduction_from_train_only / total_params)
        
        logger.info(f"Full vocabulary: {total_vocab_size} tokens")
        logger.info(f"Keeping only train tokens reduces params by {param_reduction_from_train_only:,} parameters")
        logger.info(f"This is {train_only_reduction_percent:.2f}% of total model parameters")
        
        # If train-only reduction already exceeds target, keep all train tokens
        if train_only_reduction_percent >= prune_percent:
            logger.info(f"Train-only reduction ({train_only_reduction_percent:.2f}%) already exceeds target ({prune_percent}%)")
            logger.info("Keeping all training tokens without further pruning")
            return train_token_ids, []
        
        # Otherwise, calculate how many more tokens we need to prune
        target_param_reduction = (prune_percent / 100) * total_params
        additional_param_reduction_needed = target_param_reduction - param_reduction_from_train_only
        additional_tokens_to_remove = int(additional_param_reduction_needed / embedding_dim)
        
        # Get non-special tokens from the train set
        train_non_special_tokens = [token_id for token_id in train_token_ids if token_id not in special_tokens]
        
        # We can't remove more tokens than are available in the non-special train set
        additional_tokens_to_remove = min(additional_tokens_to_remove, len(train_non_special_tokens))
        
        logger.info(f"Need additional {additional_param_reduction_needed:,} parameter reduction")
        logger.info(f"This requires removing {additional_tokens_to_remove} more tokens")
        
        # Calculate how many non-special tokens to keep
        num_non_special_to_keep = len(train_non_special_tokens) - additional_tokens_to_remove
        num_non_special_to_keep = max(min_tokens, num_non_special_to_keep)
        
        # Get tokens to keep (most frequent ones first)
        tokens_to_keep = special_tokens.copy()
        tokens_kept = 0
        
        # Add most frequent non-special tokens
        for token_id, count in sorted_tokens:
            if token_id not in special_tokens:
                tokens_to_keep.append(token_id)
                tokens_kept += 1
                if tokens_kept >= num_non_special_to_keep:
                    break
        
        # Calculate tokens removed from training set
        tokens_to_remove = [token_id for token_id in train_token_ids if token_id not in tokens_to_keep]
        
        # Calculate final parameter reduction
        token_reduction = (total_vocab_size - len(tokens_to_keep))
        final_param_reduction = token_reduction * embedding_dim
        final_reduction_percent = 100 * (final_param_reduction / total_params)
        
        logger.info(f"Kept {len(tokens_to_keep)} tokens total ({len(tokens_to_keep) - len(special_tokens)} non-special)")
        logger.info(f"Removed {len(tokens_to_remove)} tokens from training set")
        logger.info(f"Total vocabulary reduction: {token_reduction} tokens from full vocabulary")
        logger.info(f"This reduces parameters by {final_param_reduction:,} parameters")
        logger.info(f"Overall parameter reduction: {final_reduction_percent:.2f}% of total model parameters")
        
        return tokens_to_keep, tokens_to_remove
    
    else:
        # Traditional token-based pruning
        # Calculate number of tokens to keep (excluding special tokens)
        num_prunable_tokens = len(sorted_tokens) - len(special_tokens)
        num_tokens_to_keep = max(min_tokens, int(num_prunable_tokens * (1 - prune_percent / 100)))
        
        # Get tokens to keep (most frequent ones)
        tokens_to_keep = special_tokens.copy()
        
        # Add most frequent non-special tokens
        for token_id, count in sorted_tokens:
            if token_id not in special_tokens:
                tokens_to_keep.append(token_id)
                if len(tokens_to_keep) - len(special_tokens) >= num_tokens_to_keep:
                    break
        
        # Get tokens to remove
        tokens_to_remove = [token_id for token_id, _ in sorted_tokens if token_id not in tokens_to_keep]
        
        # Calculate parameter reduction for logging
        param_reduction_percent = 100 * (len(tokens_to_remove) / len(train_token_ids))
        logger.info(f"Kept {len(tokens_to_keep)} tokens, removed {len(tokens_to_remove)} tokens")
        logger.info(f"This is a {param_reduction_percent:.2f}% reduction in token count")
        
        if total_params and embedding_dim:
            param_reduction = len(tokens_to_remove) * embedding_dim
            overall_param_reduction_percent = 100 * (param_reduction / total_params)
            logger.info(f"Parameter reduction: {param_reduction:,} parameters")
            logger.info(f"Overall parameter reduction: {overall_param_reduction_percent:.2f}% of total model parameters")
    
    return tokens_to_keep, tokens_to_remove

def setup_frequency_based_model(task_name, model_name, prune_percent=0, param_based=False):
    """
    Set up a model with reduced vocabulary based on frequency-based pruning.
    This is equivalent to the clustering method but using frequency-based token selection.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens or parameters to prune based on frequency
        param_based: If True, prune based on parameter percentage rather than token percentage
        
    Returns:
        model: Model with reduced vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: None for frequency-based method
    """
    pruning_type = "parameter-based" if param_based else "token-based"
    logger.info(f"Setting up {pruning_type} frequency-based model for {task_name} with {prune_percent}% pruning")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with counts
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, all_token_ids = get_dataset_tokens_with_counts(vocab_name, train_only=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Get total model parameters and embedding dimension
    total_params = sum(p.numel() for p in model.parameters())
    embedding_dim = model.model.embeddings.tok_embeddings.embedding_dim
    total_vocab_size = model.model.embeddings.tok_embeddings.num_embeddings
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Token embeddings: {total_vocab_size:,} tokens with dimension {embedding_dim}")
    
    # Apply frequency-based pruning if requested
    if prune_percent > 0:
        tokens_to_keep, _ = frequency_based_pruning(
            token_counter, 
            prune_percent,
            param_based=param_based,
            embedding_dim=embedding_dim,
            total_params=total_params,
            total_vocab_size=total_vocab_size
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
    
    return model, token_map, None  # No OOV lookup for frequency method 