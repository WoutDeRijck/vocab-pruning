#!/usr/bin/env python
# coding: utf-8

"""
Attention-based vocabulary pruning.

This module implements vocabulary pruning based on token attention scores
from a fine-tuned model's attention patterns.
"""

import logging
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import task_to_keys, create_reduced_embeddings
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def get_token_attention_importance(task_name, model_name):
    """
    Calculate token importance based on attention for a specific GLUE task.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Model checkpoint to use for importance calculation (preferably fine-tuned)
        
    Returns:
        token_importance: Dictionary mapping token IDs to importance scores
        all_token_ids: Set of all token IDs in the dataset
    """
    logger.info(f"Calculating token attention importance for {task_name} using {model_name}")
    
    # Determine the number of labels based on the task
    task_meta = get_task_metadata(task_name)
    num_labels = task_meta["n_labels"]
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        output_attentions=True
    )
    
    # Load dataset
    raw_datasets = load_dataset("glue", task_name)
    
    # For mnli, use the train set only
    if task_name == "mnli":
        dataset = raw_datasets["train"]
    else:
        # For other tasks, combine train and validation sets
        train_dataset = raw_datasets["train"]
        validation_dataset = raw_datasets["validation"]
        dataset = concatenate_datasets([train_dataset, validation_dataset])
    
    logger.info(f"Processing dataset with {len(dataset)} examples")
    
    # Get the sentence keys
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Initialize importance dictionary for each token in the vocab
    vocab_size = len(tokenizer)
    token_importance = {token_id: 0.0 for token_id in range(vocab_size)}
    samples_seen = {token_id: 0 for token_id in range(vocab_size)}
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process batches
    batch_size = 32
    
    # Create progress bar
    progress_bar = tqdm(range(0, len(dataset), batch_size), desc="Processing batches")
    
    # Set of all token IDs seen in the dataset
    all_token_ids = set()
    
    for i in progress_bar:
        batch = dataset[i:min(i+batch_size, len(dataset))]
        
        # Prepare inputs based on task
        if sentence2_key is None:
            texts = batch[sentence1_key]
            encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        else:
            texts1 = batch[sentence1_key]
            texts2 = batch[sentence2_key]
            encodings = tokenizer(texts1, texts2, padding=True, truncation=True, return_tensors="pt")
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        # Update set of all token IDs
        all_token_ids.update(input_ids.cpu().numpy().flatten())
        
        # Forward pass with gradient calculation disabled
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_attentions=True
            )
        
        # Get attention matrices
        attentions = outputs.attentions
        
        # For each sample in the batch
        for sample_idx in range(input_ids.shape[0]):
            tokens = input_ids[sample_idx]
            mask = attention_mask[sample_idx]
            seq_len = mask.sum().item()
            
            # Only consider non-padding tokens
            valid_tokens = tokens[:seq_len]
            
            # Aggregate attention across all layers and heads
            layer_attentions = []
            for layer_idx in range(len(attentions)):
                layer_attn = attentions[layer_idx][sample_idx]
                avg_head_attn = torch.mean(layer_attn, dim=0)
                layer_attentions.append(avg_head_attn)
            
            # Average across layers
            avg_attention = torch.mean(torch.stack(layer_attentions), dim=0)
            
            # Calculate attention received by each token
            attention_received = avg_attention.sum(dim=0)[:seq_len]
            
            # Update importance scores
            for pos, token_id in enumerate(valid_tokens):
                token_id_item = token_id.item()
                token_importance[token_id_item] += attention_received[pos].item()
                samples_seen[token_id_item] += 1
        
        # Update progress bar with some stats
        if i % 50 == 0:
            progress_bar.set_postfix({
                "unique_tokens": sum(1 for freq in samples_seen.values() if freq > 0),
                "max_freq": max(samples_seen.values())
            })
    
    # Normalize by frequency
    normalized_importance = {}
    for token_id in token_importance:
        if samples_seen[token_id] > 0:
            normalized_importance[token_id] = token_importance[token_id] / samples_seen[token_id]
        else:
            normalized_importance[token_id] = 0.0
    
    logger.info(f"Found {len(all_token_ids)} unique tokens in the dataset")
    
    return normalized_importance, all_token_ids

def attention_based_pruning(token_importance, all_token_ids, prune_percent, min_tokens=5,
                         param_based=False, embedding_dim=768, total_params=None,
                         total_vocab_size=None):
    """
    Prune vocabulary based on token attention importance scores.
    
    Args:
        token_importance: Dict with token_id -> importance score mappings
        all_token_ids: Set of all token IDs seen in the dataset
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
        logger.info(f"Performing parameter-based attention pruning with prune_percent={prune_percent}% of total parameters")
    else:
        logger.info(f"Performing token-based attention pruning with prune_percent={prune_percent}%")
    
    # Sort tokens by importance (highest importance first)
    sorted_tokens = sorted(
        [(token_id, token_importance[token_id]) for token_id in all_token_ids],
        key=lambda x: x[1],
        reverse=True
    )
    
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
        
        # Get tokens to keep (highest importance ones first)
        tokens_to_keep = special_tokens.copy()
        tokens_kept = 0
        
        # Add most important non-special tokens
        for token_id, _ in sorted_tokens:
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
        # Calculate number of tokens to keep (excluding special tokens)
        num_prunable_tokens = len(sorted_tokens) - len(special_tokens)
        num_tokens_to_keep = max(min_tokens, int(num_prunable_tokens * (1 - prune_percent / 100)))
        
        # Get tokens to keep (most important ones)
        tokens_to_keep = special_tokens.copy()
        
        # Add most important non-special tokens
        for token_id, importance in sorted_tokens:
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

def setup_attention_based_model(task_name, model_name, attention_model=None, prune_percent=20, param_based=False):
    """
    Set up a model with attention-based vocabulary pruning.
    
    Args:
        task_name: Name of the GLUE task to use for pruning
        model_name: Name or path of the base model to prune
        attention_model: Pre-trained model to use for attention calculation, defaults to model_name
        prune_percent: Percentage of tokens or parameters to prune
        param_based: If True, prune based on parameter percentage rather than token percentage
        
    Returns:
        model: Model with pruned vocabulary
    """
    if param_based:
        logger.info(f"Setting up model with parameter-based attention pruning ({prune_percent}% of parameters)")
    else:
        logger.info(f"Setting up model with token-based attention pruning ({prune_percent}% of tokens)")
    
    # Set up the attention model
    attention_model_name = attention_model if attention_model is not None else model_name
    logger.info(f"Using {attention_model_name} for attention calculation")
    
    # Load GLUE task metadata
    task = get_task_metadata(task_name)
    
    # Get token importance from attention and dataset tokens
    token_importance, all_token_ids = get_token_attention_importance(task_name, attention_model_name)
    
    # Load the base model for pruning
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=task["n_labels"], cache_dir=None
    )
    
    # Get total parameters and embedding dimension for parameter-based pruning
    total_params = sum(p.numel() for p in model.parameters())
    embedding_dim = model.config.hidden_size
    total_vocab_size = model.config.vocab_size
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Vocabulary size: {total_vocab_size}")
    
    # Prune vocabulary based on token importance
    tokens_to_keep, tokens_to_remove = attention_based_pruning(
        token_importance, 
        all_token_ids, 
        prune_percent,
        param_based=param_based,
        embedding_dim=embedding_dim,
        total_params=total_params,
        total_vocab_size=total_vocab_size
    )
    
    # Create a reduced embedding layer
    model = replace_embeddings(model, tokens_to_keep)
    
    return model, dict(zip(tokens_to_keep, range(len(tokens_to_keep)))), None

# Function to create reduced embeddings from tokens_to_keep
def replace_embeddings(model, tokens_to_keep):
    """
    Replace the embedding layer with a reduced version containing only the specified tokens.
    
    Args:
        model: The model to modify
        tokens_to_keep: List of token IDs to keep in the reduced vocabulary
        
    Returns:
        The modified model with reduced embedding layer
    """
    # Create reduced embeddings
    logger.info(f"Creating reduced embeddings for {len(tokens_to_keep)} tokens")
    token_map, reduced_embeddings = create_reduced_embeddings(tokens_to_keep, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model 