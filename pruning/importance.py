"""
Word importance-based vocabulary pruning.

This module implements vocabulary pruning based on token importance using TF-IDF scores,
without OOV token clustering (OOV tokens are handled by mapping to UNK).
"""

import logging
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from .base import create_reduced_embeddings
from .importance_oov import get_dataset_tokens_with_importance, importance_based_pruning
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def setup_importance_based_model(task_name, model_name, prune_percent=20, importance_type=3):
    """
    Set up a model with word importance-based vocabulary pruning (without OOV clustering).
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune based on importance
        importance_type: Word importance setting (0=frequency only, 1-3=TF-IDF variants)
        
    Returns:
        model: Model with importance-based vocabulary pruning
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: None for this method (OOV tokens mapped to UNK)
    """
    logger.info(f"Setting up importance-based model for {task_name} with {prune_percent}% pruning, importance_type={importance_type}")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with token counts and importance scores
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, token_importance, all_token_ids = get_dataset_tokens_with_importance(vocab_name, train_only=True, importance_type=importance_type)
    
    # Load model for embedding extraction
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Importance-based pruning
    tokens_to_keep, _ = importance_based_pruning(token_counter, token_importance, prune_percent)
    
    # Create reduced embeddings (without OOV clustering)
    token_map, reduced_embeddings = create_reduced_embeddings(tokens_to_keep, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, None  # No OOV lookup, OOV tokens mapped to UNK 