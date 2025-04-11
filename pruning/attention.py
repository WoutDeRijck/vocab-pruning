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

def attention_based_pruning(token_importance, all_token_ids, prune_percent, min_tokens=5):
    """
    Prune vocabulary based on token attention importance scores.
    
    Args:
        token_importance: Dict with token_id -> importance score mappings
        all_token_ids: Set of all token IDs seen in the dataset
        prune_percent: Percentage of tokens to prune
        min_tokens: Minimum number of tokens to keep
        
    Returns:
        tokens_to_keep: List of token IDs to keep
        tokens_to_remove: List of token IDs to remove
    """
    logger.info(f"Performing attention-based pruning with prune_percent={prune_percent}%")
    
    # Sort tokens by importance (highest first)
    sorted_tokens = sorted(
        [(token_id, token_importance[token_id]) for token_id in all_token_ids],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Always keep special tokens (first 5 tokens are usually special tokens in ModernBERT)
    special_tokens = list(range(5))
    
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
    
    logger.info(f"Kept {len(tokens_to_keep)} tokens, removed {len(tokens_to_remove)} tokens")
    
    return tokens_to_keep, tokens_to_remove

def setup_attention_based_model(task_name, model_name, attention_model=None, prune_percent=20):
    """
    Set up a model with attention-based vocabulary pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        attention_model: Fine-tuned model to use for attention calculation (defaults to model_name)
        prune_percent: Percentage of tokens to prune based on attention importance
        
    Returns:
        model: Model with attention-based pruned vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: None for attention-based method (OOV tokens mapped to UNK)
    """
    logger.info(f"Setting up attention-based model for {task_name} with {prune_percent}% pruning")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # If no attention model is specified, use the base model
    if attention_model is None:
        attention_model = model_name
        logger.info(f"No specific attention model provided, using base model: {model_name}")
    else:
        logger.info(f"Using fine-tuned model for attention calculation: {attention_model}")
    
    # Get dataset vocabulary with token importance scores based on attention
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_importance, all_token_ids = get_token_attention_importance(vocab_name, attention_model)
    
    # Print top 20 most important tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    top_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("Top 20 most important tokens by attention:")
    for token_id, importance in top_tokens:
        token = tokenizer.convert_ids_to_tokens(token_id)
        logger.info(f"Token: {token}, ID: {token_id}, Importance: {importance:.6f}")
    
    # Load model for embedding extraction
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Attention-based pruning
    tokens_to_keep, _ = attention_based_pruning(token_importance, all_token_ids, prune_percent)
    
    # Create reduced embeddings
    token_map, reduced_embeddings = create_reduced_embeddings(tokens_to_keep, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, None  # No OOV lookup, OOV tokens mapped to UNK 