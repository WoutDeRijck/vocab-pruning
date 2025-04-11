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

def importance_based_pruning(token_counter, token_importance, prune_percent, min_tokens=5):
    """
    Prune vocabulary based on token importance scores.
    Used by the word importance based pruning method.
    
    Args:
        token_counter: Counter with token frequencies
        token_importance: Dict with token_id -> importance score mappings
        prune_percent: Percentage of tokens to prune
        min_tokens: Minimum number of tokens to keep
        
    Returns:
        tokens_to_keep: List of token IDs to keep
        tokens_to_remove: List of token IDs to remove
    """
    logger.info(f"Performing importance-based pruning with prune_percent={prune_percent}%")
    
    # Sort tokens by importance (highest first)
    sorted_tokens = sorted(
        [(token_id, token_importance[token_id]) for token_id in token_importance],
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

def setup_importance_oov_model(task_name, model_name, prune_percent=20, num_clusters=50, importance_type=3):
    """
    Set up a model with word importance-based OOV vocabulary pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune based on importance
        num_clusters: Number of clusters for OOV token mapping
        importance_type: Word importance setting (0=frequency only, 1-3=TF-IDF variants)
        
    Returns:
        model: Model with importance-OOV based vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping from OOV token ID to cluster representative ID
    """
    logger.info(f"Setting up importance-OOV model for {task_name} with {prune_percent}% pruning, {num_clusters} OOV clusters, importance_type={importance_type}")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with token counts and importance scores
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, token_importance, all_token_ids = get_dataset_tokens_with_importance(vocab_name, train_only=True, importance_type=importance_type)
    
    # Load model for embedding extraction
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Step 1: Importance-based pruning
    if importance_type > 0 and NLTK_AVAILABLE:
        tokens_to_keep, tokens_to_remove = importance_based_pruning(token_counter, token_importance, prune_percent)
    else:
        tokens_to_keep, tokens_to_remove = frequency_based_pruning(token_counter, prune_percent)
    
    # Step 2: Cluster removed tokens
    oov_token_map, cluster_centers = cluster_removed_tokens(tokens_to_remove, model, num_clusters)
    
    # Step 3: Create hybrid embeddings
    token_map, reduced_embeddings, oov_lookup = create_hybrid_embeddings(
        tokens_to_keep, oov_token_map, cluster_centers, model
    )
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, oov_lookup 