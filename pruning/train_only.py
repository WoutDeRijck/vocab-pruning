"""
Train-only vocabulary pruning.

This module implements a simple pruning technique that keeps 
all tokens that appear in the training set without any additional pruning.
"""

import logging
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .base import get_dataset_tokens_with_counts, create_reduced_embeddings
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def setup_train_only_model(task_name, model_name):
    """
    Set up a model with vocabulary limited to only tokens that appear in the training set.
    No pruning is applied beyond keeping only the training set vocabulary.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        
    Returns:
        model: Model with reduced vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: None for train-only method
    """
    logger.info(f"Setting up train-only model for {task_name}")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with counts from training set only
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, all_token_ids = get_dataset_tokens_with_counts(vocab_name, train_only=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Use all tokens found in the training set (no further pruning)
    tokens_to_keep = sorted(list(all_token_ids))
    
    # Create reduced embeddings
    token_map, reduced_embeddings = create_reduced_embeddings(tokens_to_keep, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    # Log vocabulary reduction statistics
    original_vocab_size = model.config.vocab_size
    reduced_vocab_size = len(tokens_to_keep)
    reduction_percent = (1 - reduced_vocab_size / original_vocab_size) * 100
    
    logger.info(f"Original vocabulary size: {original_vocab_size}")
    logger.info(f"Reduced vocabulary size: {reduced_vocab_size}")
    logger.info(f"Vocabulary reduction: {reduction_percent:.2f}%")
    
    return model, token_map, None  # No OOV lookup for train-only method 