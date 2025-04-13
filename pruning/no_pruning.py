"""
No-pruning baseline.

This module implements a baseline method that keeps the original vocabulary
without any pruning for comparison with pruning methods.
"""

import logging
import torch
from transformers import AutoModelForSequenceClassification

from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def setup_no_pruning_model(task_name, model_name):
    """
    Set up a model without any vocabulary pruning for baseline comparison.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        
    Returns:
        model: Original model without vocabulary changes
        token_map: None (no mapping, identity mapping is used implicitly)
        oov_lookup: None (no OOV handling)
    """
    logger.info(f"Setting up no-pruning baseline model for {task_name}")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Load model without any vocabulary modification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Log that we're using the original vocabulary
    logger.info(f"Using original vocabulary with size: {model.config.vocab_size}")
    logger.info("No vocabulary pruning applied (baseline method)")
    
    # Return the model with no token mapping
    return model, None, None 