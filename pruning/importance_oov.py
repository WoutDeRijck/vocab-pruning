"""
Word importance-based OOV vocabulary pruning.

This module implements vocabulary pruning based on token importance using TF-IDF scores,
combined with OOV token clustering.
"""

import logging
import numpy as np
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

try:
    import nltk
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from .base import create_hybrid_embeddings
from .frequency import frequency_based_pruning
from .frequency_oov import cluster_removed_tokens
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def get_tfidf_vector(texts, importance_type=3):
    """
    Calculate TF-IDF scores for tokens in texts.
    Used for word importance based pruning.
    
    Args:
        texts: List of text strings
        importance_type: Word importance setting (1=no norm, 2=L1 norm, 3=L2 norm)
        
    Returns:
        tfidf_vectorizer: Fitted TF-IDF vectorizer
    """
    if not NLTK_AVAILABLE:
        logger.warning("NLTK not available, using frequency-based pruning instead")
        return None
        
    # Download stopwords if not already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Configure TF-IDF vectorizer based on importance_type parameter
    if importance_type == 1:
        # No normalization
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            norm=None
        )
    elif importance_type == 2:
        # L1 normalization
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            norm='l1'
        )
    else:
        # Default L2 normalization
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english')
        )
    
    # Fit the vectorizer on the texts
    tfidf_vectorizer.fit(texts)
    
    return tfidf_vectorizer

def get_dataset_tokens_with_importance(task_name, train_only=True, importance_type=0):
    """
    Extract vocabulary from a GLUE task dataset with token importance scores.
    Used for word importance based pruning.
    
    Args:
        task_name: Name of the GLUE task
        train_only: Whether to use only the training set
        importance_type: Word importance setting (0=frequency only, 1-3=TF-IDF variants)
        
    Returns:
        token_counter: Counter with token_id -> count mappings
        token_importance: Dict with token_id -> importance score mappings
        all_token_ids: Set of all token IDs seen in the dataset
    """
    logger.info(f"Getting dataset vocabulary with importance scores for {task_name}, train_only={train_only}, importance_type={importance_type}")
    
    # Import task_to_keys from base
    from .base import task_to_keys
    
    # Load the dataset
    from datasets import load_dataset
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Counter to store token counts
    from collections import Counter
    token_counter = Counter()
    
    # Set to store all unique token IDs
    all_token_ids = set()
    
    # Dictionary to collect all texts for TF-IDF calculation
    all_texts = []
    token_to_word = {}  # Map from token_id to original word
    
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
            # Add text for TF-IDF calculation
            all_texts.append(text)
            
            # Get token IDs
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            token_counter.update(token_ids)
            all_token_ids.update(token_ids)
            
            # Map token IDs to words for later TF-IDF lookup
            tokens = tokenizer.tokenize(text)
            token_ids_no_special = tokenizer.convert_tokens_to_ids(tokens)
            
            for token_id, token in zip(token_ids_no_special, tokens):
                token_to_word[token_id] = token.replace("##", "")  # Remove subword prefixes
        
        # Process sentence2 if it exists
        if sentence2_key is not None:
            texts = raw_datasets[split][sentence2_key]
            for text in texts:
                if text is not None:
                    all_texts.append(text)
                    
                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                    token_counter.update(token_ids)
                    all_token_ids.update(token_ids)
                    
                    tokens = tokenizer.tokenize(text)
                    token_ids_no_special = tokenizer.convert_tokens_to_ids(tokens)
                    
                    for token_id, token in zip(token_ids_no_special, tokens):
                        token_to_word[token_id] = token.replace("##", "")
    
    # Initialize token_importance with frequency counts by default
    token_importance = {token_id: count for token_id, count in token_counter.items()}
    
    # If TF-IDF is requested, calculate word importance scores
    if importance_type > 0 and NLTK_AVAILABLE:
        logger.info(f"Calculating TF-IDF scores with importance_type={importance_type}")
        
        # Train TF-IDF vectorizer on all texts
        tfidf_vectorizer = get_tfidf_vector(all_texts, importance_type)
        
        if tfidf_vectorizer:
            # Get feature names (words) and their index in the TF-IDF matrix
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Calculate TF-IDF scores for all texts
            tfidf_matrix = tfidf_vectorizer.transform(all_texts)
            
            # Get average TF-IDF score for each word across all documents
            avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            # Create a dictionary of word -> average TF-IDF score
            word_importance = {word: score for word, score in zip(feature_names, avg_tfidf)}
            
            # Map token IDs to TF-IDF scores using token_to_word mapping
            for token_id, word in token_to_word.items():
                if word in word_importance:
                    # Use TF-IDF score as importance
                    token_importance[token_id] = word_importance[word]
                else:
                    # Keep frequency count for tokens not in TF-IDF vocabulary
                    token_importance[token_id] = token_counter[token_id] if token_id in token_counter else 0
    
    logger.info(f"Found {len(all_token_ids)} unique tokens in the dataset")
    
    return token_counter, token_importance, all_token_ids

def importance_based_pruning(token_counter, token_importance, prune_percent, min_tokens=5,
                          param_based=False, embedding_dim=768, total_params=None,
                          total_vocab_size=None):
    """
    Prune vocabulary based on token importance scores.
    Used by the word importance based pruning method.
    
    Args:
        token_counter: Counter with token frequencies
        token_importance: Dict with token_id -> importance score mappings
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
        logger.info(f"Performing parameter-based importance pruning with prune_percent={prune_percent}% of total parameters")
    else:
        logger.info(f"Performing token-based importance pruning with prune_percent={prune_percent}%")
    
    # Sort tokens by importance (highest importance first)
    sorted_tokens = sorted(
        token_importance.items(), 
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
        for token_id, _ in sorted_tokens:
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

def setup_importance_oov_model(task_name, model_name, prune_percent=20, num_clusters=50, importance_type=3, param_based=False):
    """
    Set up a model with word importance-based vocabulary pruning and OOV clustering.
    This combines approaches from frequency-based pruning and clustering.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens or parameters to prune based on importance
        num_clusters: Number of clusters to create for OOV tokens
        importance_type: Word importance setting (0=frequency only, 1-3=TF-IDF variants)
        param_based: If True, prune based on parameter percentage rather than token percentage
        
    Returns:
        model: Model with importance-based vocabulary pruning and OOV clustering
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping for OOV tokens to cluster representatives
    """
    pruning_type = "parameter-based" if param_based else "token-based"
    logger.info(f"Setting up {pruning_type} importance-OOV model for {task_name} with {prune_percent}% pruning, {num_clusters} OOV clusters, importance_type={importance_type}")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with token counts and importance scores
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, token_importance, all_token_ids = get_dataset_tokens_with_importance(vocab_name, train_only=True, importance_type=importance_type)
    
    # Load model for embedding extraction
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Get total model parameters and embedding dimension
    total_params = sum(p.numel() for p in model.parameters())
    embedding_dim = model.model.embeddings.tok_embeddings.embedding_dim
    total_vocab_size = model.model.embeddings.tok_embeddings.num_embeddings
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Token embeddings: {total_vocab_size:,} tokens with dimension {embedding_dim}")
    
    # Importance-based pruning
    tokens_to_keep, tokens_to_remove = importance_based_pruning(
        token_counter,
        token_importance,
        prune_percent,
        param_based=param_based,
        embedding_dim=embedding_dim,
        total_params=total_params,
        total_vocab_size=total_vocab_size
    )
    
    # Cluster removed tokens if there are any tokens to remove
    if tokens_to_remove:
        oov_lookup = cluster_removed_tokens(tokens_to_remove, tokens_to_keep, model, num_clusters)
    else:
        oov_lookup = {}
    
    # Create hybrid embeddings with OOV token clustering
    logger.info(f"Creating hybrid embeddings for {len(tokens_to_keep)} kept tokens and {len(oov_lookup)} OOV tokens")
    token_map, reduced_embeddings = create_hybrid_embeddings(tokens_to_keep, oov_lookup, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, oov_lookup 