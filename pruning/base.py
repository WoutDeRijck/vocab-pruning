"""
Base utilities for vocabulary pruning.

This module contains common functions and utilities used by all pruning methods.
"""

import logging
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn as nn
import copy

# Configure logging
logger = logging.getLogger(__name__)

# GLUE task input keys mapping
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Add a new TokenMappingWrapper class to handle token remapping
class TokenMappingWrapper(nn.Module):
    """
    Model wrapper that applies token mapping before the embedding layer
    to handle vocabulary pruning.
    """
    def __init__(self, model, token_map=None, oov_lookup=None):
        super().__init__()
        self.model = model
        self.token_map = token_map or {}
        self.oov_lookup = oov_lookup or {}
        
        # Combine mappings for faster lookup
        self.combined_map = {}
        if token_map:
            self.combined_map.update(token_map)
        if oov_lookup:
            self.combined_map.update(oov_lookup)
            
        logger.info(f"TokenMappingWrapper initialized with {len(self.token_map)} mapped tokens and {len(set(self.oov_lookup.values()))} OOV clusters")
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Only process if we have input_ids and mappings
        if input_ids is not None and self.combined_map:
            # Create a new tensor to hold the mapped input IDs
            mapped_input_ids = torch.zeros_like(input_ids)
            
            # Apply mapping to each element in the input_ids tensor
            for i in range(input_ids.size(0)):
                for j in range(input_ids.size(1)):
                    token_id = input_ids[i, j].item()
                    if token_id in self.combined_map:
                        mapped_input_ids[i, j] = self.combined_map[token_id]
                    else:
                        # For OOV tokens not in mapping, use UNK token (0)
                        mapped_input_ids[i, j] = 0
            
            # Replace input_ids with mapped version
            input_ids = mapped_input_ids
        
        # Forward pass with mapped input_ids
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    # Forward all other attributes to the wrapped model
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

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
    logger.info(f"Getting dataset vocabulary for {task_name}, train_only={train_only}")
    
    # Load the dataset
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Counter to store token counts
    token_counter = Counter()
    
    # Set to store all unique token IDs
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
def create_hybrid_embeddings(tokens_to_keep, oov_lookup, model):
    """
    Create hybrid embeddings for a reduced vocabulary.
    Used by multiple pruning methods.
    
    Args:
        tokens_to_keep: List of token IDs to keep from original vocab
        oov_lookup: Dictionary mapping OOV token IDs to representative token IDs
        model: Original model with full embedding matrix
        
    Returns:
        token_map: Mapping from original token ID to new token ID
        reduced_embeddings: Tensor with reduced vocabulary embeddings
    """
    # Get original embedding matrix
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    
    # Create a map from original token ID to new token ID
    token_map = {}
    for i, token_id in enumerate(tokens_to_keep):
        token_map[token_id] = i
    
    # Get the embeddings for tokens we're keeping
    kept_embeddings = original_embeddings[tokens_to_keep]
    
    # For OOV, get representative embeddings
    if oov_lookup:
        # Get unique representatives for logging
        oov_clusters = set(oov_lookup.values())
        logger.info(f"Creating embeddings with {len(tokens_to_keep)} kept tokens and {len(oov_clusters)} OOV representatives")
        
        # The OOV map already points to original token IDs that are in tokens_to_keep
        # So we just need to update the map to point to the new indices
        for token_id in oov_lookup:
            if token_id not in token_map:  # Skip if already kept
                representative_id = oov_lookup[token_id]
                if representative_id in token_map:
                    # Map to existing representative in reduced vocabulary
                    token_map[token_id] = token_map[representative_id]
                else:
                    # This shouldn't happen with proper cluster setup
                    logger.warning(f"Representative token {representative_id} for {token_id} not in token_map!")
                    # Fallback to UNK token
                    token_map[token_id] = 0
    else:
        logger.info(f"Creating embeddings with {len(tokens_to_keep)} kept tokens (no OOV mapping)")
    
    return token_map, kept_embeddings 