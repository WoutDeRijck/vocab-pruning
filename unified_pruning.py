#!/usr/bin/env python
# coding: utf-8

"""
Unified Vocabulary Pruning Script

This script combines multiple vocabulary pruning techniques:
1. Clustering-based pruning: Group similar tokens and keep representatives
2. Frequency-based pruning: Remove least frequently used tokens
3. Hybrid pruning: Combine frequency-based pruning with clustering for OOV tokens
4. Word importance pruning: Use TF-IDF to determine token importance

For GLUE benchmark tasks.
"""

import os
import argparse
import logging
from functools import partial
from collections import Counter
import copy
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist, pdist, squareform
from evaluate import load

# Optional TF-IDF imports for word importance
try:
    import nltk
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Disable PyTorch dynamo compiler to avoid CUDA illegal memory access errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Mapping of GLUE tasks to their input keys
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# GLUE task metadata
glue_tasks = {
    "cola": {
        "abbr": "CoLA",
        "name": "Corpus of Linguistic Acceptability",
        "description": "Predict whether a sequence is a grammatical English sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Misc.",
        "size": "8.5k",
        "metrics": "Matthews corr.",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [matthews_corrcoef],
        "n_labels": 2,
    },
    "sst2": {
        "abbr": "SST-2",
        "name": "Stanford Sentiment Treebank",
        "description": "Predict the sentiment of a given sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Movie reviews",
        "size": "67k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "mrpc": {
        "abbr": "MRPC",
        "name": "Microsoft Research Paraphrase Corpus",
        "description": "Predict whether two sentences are semantically equivalent",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "News",
        "size": "3.7k",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score, f1_score],
        "n_labels": 2,
    },
    "stsb": {
        "abbr": "SST-B",
        "name": "Semantic Textual Similarity Benchmark",
        "description": "Predict the similarity score for two sentences on a scale from 1 to 5",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Misc.",
        "size": "7k",
        "metrics": "Pearson/Spearman corr.",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [pearsonr, spearmanr],
        "n_labels": 1,
    },
    "qqp": {
        "abbr": "QQP",
        "name": "Quora question pair",
        "description": "Predict if two questions are a paraphrase of one another",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Social QA questions",
        "size": "364k",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["question1", "question2"],
        "target": "label",
        "metric_funcs": [f1_score, accuracy_score],
        "n_labels": 2,
    },
    "mnli-matched": {
        "abbr": "MNLI",
        "name": "Mulit-Genre Natural Language Inference",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation_matched", "test": "test_matched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 3,
    },
    "mnli-mismatched": {
        "abbr": "MNLI",
        "name": "Mulit-Genre Natural Language Inference",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation_mismatched", "test": "test_mismatched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 3,
    },
    "qnli": {
        "abbr": "QNLI",
        "name": "Stanford Question Answering Dataset",
        "description": "Predict whether the context sentence contains the answer to the question",
        "task_type": "Inference Tasks",
        "domain": "Wikipedia",
        "size": "105k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["question", "sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "rte": {
        "abbr": "RTE",
        "name": "Recognize Textual Entailment",
        "description": "Predict whether one sentece entails another",
        "task_type": "Inference Tasks",
        "domain": "News, Wikipedia",
        "size": "2.5k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "wnli": {
        "abbr": "WNLI",
        "name": "Winograd Schema Challenge",
        "description": "Predict if the sentence with the pronoun substituted is entailed by the original sentence",
        "task_type": "Inference Tasks",
        "domain": "Fiction books",
        "size": "634",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models with unified vocabulary pruning"
    )
    
    # Common arguments
    parser.add_argument(
        "--task", 
        type=str, 
        default="sst2", 
        choices=list(task_to_keys.keys()),
        help="GLUE task name"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name or path"
    )
    
    parser.add_argument(
        "--pruning_method", 
        type=str, 
        default="clustering",
        choices=["clustering", "frequency", "hybrid", "importance"],
        help="Method to use for vocabulary pruning"
    )
    
    parser.add_argument(
        "--prune_percent", 
        type=float, 
        default=20,
        help="Percentage of vocabulary to prune"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=8e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=8e-6,
        help="Weight decay"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Training batch size"
    )
    
    # Clustering method specific arguments
    parser.add_argument(
        "--clustering_method", 
        type=str, 
        default="agglomerative",
        choices=["agglomerative", "kmeans"],
        help="Clustering algorithm (for clustering method)"
    )
    
    # Hybrid method specific arguments
    parser.add_argument(
        "--num_clusters", 
        type=int, 
        default=50,
        help="Number of clusters for OOV token mapping (for hybrid method)"
    )
    
    # Word importance specific arguments
    parser.add_argument(
        "--importance_type", 
        type=int, 
        default=3,
        choices=[0, 1, 2, 3],
        help="Word importance calculation type: 0=off, 1=no norm, 2=L1 norm, 3=L2 norm (default)"
    )
    
    # Custom split related arguments
    parser.add_argument(
        "--use_custom_splits", 
        action="store_true",
        help="Use custom train/validation/test splits instead of original GLUE splits"
    )
    
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.8,
        help="Ratio of data to use for training when using custom splits"
    )
    
    parser.add_argument(
        "--validation_ratio", 
        type=float, 
        default=0.1,
        help="Ratio of data to use for validation when using custom splits"
    )
    
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.1,
        help="Ratio of data to use for testing when using custom splits"
    )
    
    # Common optional arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./unified_model_output",
        help="Output directory for model checkpoints and logs"
    )
    
    parser.add_argument(
        "--train_only", 
        action="store_true",
        help="Use only the training set for vocabulary extraction"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args() 

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
    logger.info(f"Getting dataset vocabulary with counts for {task_name}, train_only={train_only}")
    
    # Load the dataset
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Counter to store token counts
    token_counter = Counter()
    
    # Set to store all unique token IDs (including low frequency)
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
            # Get token IDs instead of tokens
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
    
    # Load the dataset
    raw_datasets = load_dataset("glue", task_name)
    sentence1_key, sentence2_key = task_to_keys[task_name]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Counter to store token counts
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

def cluster_embeddings(token_ids, model, prune_percent, clustering_method="agglomerative"):
    """
    Cluster token embeddings and prune vocabulary based on similarity.
    Used by the clustering-based pruning method.
    
    Args:
        token_ids: List of token IDs to consider for clustering
        model: Model whose embeddings will be used
        prune_percent: Percentage of vocabulary to prune through clustering
        clustering_method: Method to use for clustering (agglomerative or kmeans)
        
    Returns:
        List of token IDs to keep after pruning
    """
    logger.info(f"Clustering embeddings for {len(token_ids)} tokens with {prune_percent}% pruning using {clustering_method}")
    
    # Get original embedding matrix
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    
    # Get embeddings for tokens in the vocabulary
    token_embeddings = original_embeddings[token_ids]
    embeddings_np = token_embeddings.cpu().numpy()
    
    # Always keep special tokens (usually the first few tokens)
    # For ModernBERT, assume the first 5 tokens are special
    special_tokens = token_ids[:5]
    prunable_tokens = token_ids[5:]
    prunable_embeddings = original_embeddings[prunable_tokens].cpu().numpy()
    
    # Calculate number of clusters
    n_clusters = len(prunable_tokens) - int(len(token_ids) * prune_percent / 100)
    logger.info(f"Clustering into {n_clusters} clusters (reducing from {len(prunable_tokens)} tokens)")
    
    # Normalize the embeddings for cosine similarity
    norms = np.linalg.norm(prunable_embeddings, axis=1, keepdims=True)
    normalized_embeddings = prunable_embeddings / norms
    
    # Perform clustering based on selected method
    logger.info(f"Performing {clustering_method} clustering...")
    if clustering_method == "kmeans":
        # Use K-means clustering
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering.fit_predict(normalized_embeddings)
    else:
        # Use agglomerative clustering
        try:
            # Try with scikit-learn's standard approach
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(normalized_embeddings)
        except TypeError:
            # If that fails, try with precomputed distances
            logger.info("Falling back to precomputed distance matrix approach")
            distances = pdist(normalized_embeddings, metric='cosine')
            distance_matrix = squareform(distances)
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
    
    # For each cluster, find the token closest to the centroid
    tokens_to_keep = []
    
    # Load tokenizer for logging token examples
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Log cluster size distribution
    cluster_sizes = {}
    for cluster_id in range(n_clusters):
        cluster_size = np.sum(cluster_labels == cluster_id)
        if cluster_size in cluster_sizes:
            cluster_sizes[cluster_size] += 1
        else:
            cluster_sizes[cluster_size] = 1
    
    # Sort and log cluster size stats
    logger.info("Cluster size distribution:")
    for size, count in sorted(cluster_sizes.items()):
        logger.info(f"  {size} tokens: {count} clusters")
    
    # Process each cluster and log examples of merged tokens
    sample_clusters_to_log = 5
    logged_clusters = 0
    
    for cluster_id in range(n_clusters):
        # Get indices of tokens in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) == 1:
            # If only one token in the cluster, keep it
            tokens_to_keep.append(prunable_tokens[cluster_indices[0]])
        else:
            # Calculate cluster centroid
            cluster_embeddings = prunable_embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find token closest to centroid
            distances_to_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances_to_centroid)]
            
            # Add the representative token to our list
            tokens_to_keep.append(prunable_tokens[closest_idx])
            
            # Log example clusters if they have multiple tokens and we haven't logged enough yet
            if len(cluster_indices) > 1 and logged_clusters < sample_clusters_to_log:
                example_tokens = [tokenizer.convert_ids_to_tokens(prunable_tokens[idx]) for idx in cluster_indices]
                kept_token = tokenizer.convert_ids_to_tokens(prunable_tokens[closest_idx])
                logger.info(f"Cluster example #{logged_clusters+1}: Kept '{kept_token}', merged with {example_tokens}")
                logged_clusters += 1
    
    # Combine special tokens with representative tokens from clusters
    final_vocab = special_tokens + tokens_to_keep
    logger.info(f"Final vocabulary size after clustering: {len(final_vocab)} (reduced from {len(token_ids)}, {(1 - len(final_vocab)/len(token_ids))*100:.2f}% reduction)")
    
    return final_vocab

def frequency_based_pruning(token_counter, prune_percent, min_tokens=5):
    """
    Prune vocabulary based on token frequency.
    Used by frequency-based and hybrid pruning methods.
    
    Args:
        token_counter: Counter with token frequencies
        prune_percent: Percentage of tokens to prune
        min_tokens: Minimum number of tokens to keep
        
    Returns:
        tokens_to_keep: List of token IDs to keep
        tokens_to_remove: List of token IDs to remove
    """
    logger.info(f"Performing frequency-based pruning with prune_percent={prune_percent}%")
    
    # Sort tokens by frequency (most common first)
    sorted_tokens = token_counter.most_common()
    
    # Always keep special tokens (first 5 tokens are usually special tokens in ModernBERT)
    special_tokens = list(range(5))
    
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
    
    logger.info(f"Kept {len(tokens_to_keep)} tokens, removed {len(tokens_to_remove)} tokens")
    
    return tokens_to_keep, tokens_to_remove

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

def cluster_removed_tokens(tokens_to_remove, model, num_clusters=50):
    """
    Cluster removed tokens to create mapping for OOV tokens.
    Used by hybrid pruning methods.
    
    Args:
        tokens_to_remove: List of token IDs that were removed
        model: Model whose embeddings will be used
        num_clusters: Number of clusters to create
        
    Returns:
        oov_token_map: Mapping from removed token ID to nearest cluster center token ID
        cluster_centers: List of token IDs representing cluster centers
    """
    logger.info(f"Clustering {len(tokens_to_remove)} removed tokens into {num_clusters} clusters")
    
    # Get original embedding matrix
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    
    # Get embeddings for removed tokens
    removed_embeddings = original_embeddings[tokens_to_remove].cpu().numpy()
    
    # Adjust number of clusters if we have fewer tokens than requested clusters
    actual_num_clusters = min(num_clusters, len(tokens_to_remove))
    
    # Perform KMeans clustering (faster than agglomerative for large number of tokens)
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(removed_embeddings)
    
    # Initialize mapping from removed token to nearest cluster center token
    oov_token_map = {}
    cluster_centers = []
    
    # For each cluster, find the token closest to the centroid
    for cluster_id in range(actual_num_clusters):
        # Get indices of tokens in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) > 0:
            # Calculate distances to centroid
            cluster_tokens = [tokens_to_remove[i] for i in cluster_indices]
            cluster_embeddings = original_embeddings[cluster_tokens].cpu().numpy()
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Find token closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            center_token_id = tokens_to_remove[closest_idx]
            
            # Map all tokens in this cluster to the center token
            for idx in cluster_indices:
                oov_token_map[tokens_to_remove[idx]] = center_token_id
            
            cluster_centers.append(center_token_id)
    
    # Load tokenizer for logging example clusters
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Log some example clusters
    logger.info("Example cluster mappings:")
    clusters_to_log = min(5, actual_num_clusters)
    
    for i in range(clusters_to_log):
        cluster_id = i
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            center_token = tokenizer.convert_ids_to_tokens(cluster_centers[i])
            example_tokens = [tokenizer.convert_ids_to_tokens(tokens_to_remove[idx]) for idx in cluster_indices[:5]]
            logger.info(f"Cluster {i+1}: Center '{center_token}', maps tokens like {example_tokens}")
    
    logger.info(f"Created {len(cluster_centers)} cluster representatives for OOV token mapping")
    
    return oov_token_map, cluster_centers 

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

def create_hybrid_embeddings(tokens_to_keep, oov_token_map, cluster_centers, model):
    """
    Create reduced embedding matrix with hybrid token mapping.
    Used by hybrid and importance-based pruning.
    
    Args:
        tokens_to_keep: List of token IDs to keep
        oov_token_map: Mapping from removed token ID to cluster center token ID
        cluster_centers: List of token IDs for cluster centers
        model: Model whose embeddings will be used
        
    Returns:
        token_map: Mapping from original token IDs to new consecutive IDs
        reduced_embeddings: Reduced embedding matrix
        oov_lookup: Lookup table for mapping OOV tokens
    """
    logger.info(f"Creating hybrid embeddings with {len(tokens_to_keep)} kept tokens and {len(cluster_centers)} OOV clusters")
    
    # Get original embedding matrix
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    
    # Combine kept tokens and cluster centers (avoiding duplicates)
    all_tokens_to_keep = list(tokens_to_keep)
    for center in cluster_centers:
        if center not in all_tokens_to_keep:
            all_tokens_to_keep.append(center)
    
    # Create mapping from original token IDs to new consecutive IDs
    token_map = {old_id: new_id for new_id, old_id in enumerate(all_tokens_to_keep)}
    
    # Create reduced embedding matrix with only the needed vectors
    reduced_embeddings = torch.stack([original_embeddings[i] for i in all_tokens_to_keep])
    
    # Create OOV lookup that maps original token IDs to their cluster representatives
    oov_lookup = {}
    for orig_id, center_id in oov_token_map.items():
        if center_id in token_map:  # Ensure the center ID is in our vocabulary
            oov_lookup[orig_id] = token_map[center_id]
    
    return token_map, reduced_embeddings, oov_lookup

class ReducedVocabDataCollator:
    """
    Data collator for reduced vocabulary models.
    Used by clustering-based pruning.
    Handles padding and preparing batches for the model.
    """
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        
    def __call__(self, features):
        # Get max length in batch
        max_length = max(len(x['input_ids']) for x in features)
        
        # Pad input_ids and attention masks
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            # Pad input_ids
            padded = f['input_ids'] + [self.pad_token_id] * (max_length - len(f['input_ids']))
            input_ids.append(padded)
            
            # Pad attention mask
            mask = f['attention_mask'] + [0] * (max_length - len(f['attention_mask']))
            attention_mask.append(mask)
            
            # Get labels
            labels.append(f['labels'])
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }

class HybridCollator:
    """
    Data collator for hybrid vocabulary models.
    Used by hybrid and importance-based pruning.
    Handles padding and remapping OOV tokens to appropriate clusters.
    """
    def __init__(self, pad_token_id, unk_token_id, oov_lookup=None):
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.oov_lookup = oov_lookup or {}
        
    def __call__(self, features):
        # Get max length in batch
        max_length = max(len(x['input_ids']) for x in features)
        
        # Pad input_ids and attention masks
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            # Pad input_ids
            padded = f['input_ids'] + [self.pad_token_id] * (max_length - len(f['input_ids']))
            input_ids.append(padded)
            
            # Pad attention mask
            mask = f['attention_mask'] + [0] * (max_length - len(f['attention_mask']))
            attention_mask.append(mask)
            
            # Get labels
            labels.append(f['labels'])
        
        # Check if any labels are -1 (test set)
        has_invalid_labels = any(label == -1 for label in labels)
        
        result = {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }
        
        # If we have invalid labels, add a flag to indicate prediction-only mode
        if has_invalid_labels:
            result['prediction_only'] = True
            
        return result

class MetricsCallback(TrainerCallback):
    """Callback to track training and evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics collection."""
        self.training_history = {
            "train": {
                "loss": [],
                "learning_rate": [],
                "epoch": [],
            },
            "eval": {
                "loss": [],
                "epoch": [],
            }
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Collect metrics when logged."""
        if logs is None:
            return
        
        logs = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        
        # Determine if train or eval metrics
        is_eval = any(k.startswith("eval_") for k in logs.keys())
        
        if is_eval:
            # Extract and store eval metrics
            for key, value in logs.items():
                if key.startswith("eval_"):
                    metric_name = key[5:]  # Remove 'eval_' prefix
                    if metric_name not in self.training_history["eval"]:
                        self.training_history["eval"][metric_name] = []
                    self.training_history["eval"][metric_name].append(value)
                elif key == "epoch":
                    self.training_history["eval"]["epoch"].append(value)
        else:
            # Extract and store training metrics
            for key, value in logs.items():
                if key in self.training_history["train"]:
                    self.training_history["train"][key].append(value) 

def compute_metrics(eval_pred, task_name):
    """
    Compute metrics for model evaluation using official GLUE metrics.
    
    Args:
        eval_pred: Tuple of predictions and labels
        task_name: Name of the GLUE task
        
    Returns:
        results: Dictionary of metric results
    """
    predictions, labels = eval_pred
    
    # Handle regression task (e.g., STS-B)
    if len(predictions.shape) == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze()
    # Handle classification task
    elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Load the official GLUE metric for this task
    metric = load('glue', task_name)
    
    # Compute metrics
    # Only compute if we have valid labels (not -1)
    valid_indices = labels != -1
    if valid_indices.any():
        results = metric.compute(predictions=predictions[valid_indices], 
                               references=labels[valid_indices])
    else:
        logger.info("No valid labels found (test set?), skipping metric computation")
        results = {}
    
    return results

def count_model_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        counts: Dictionary with parameter counts
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Try to identify the embedding layer(s)
    embedding_params = 0
    
    # For ModernBERT model structure
    if hasattr(model, 'model') and hasattr(model.model, 'embeddings'):
        if hasattr(model.model.embeddings, 'tok_embeddings'):
            embedding_params += model.model.embeddings.tok_embeddings.weight.numel()
        # Check for other embedding types like position embeddings
        for name, param in model.model.embeddings.named_parameters():
            if name != 'tok_embeddings.weight' and 'embed' in name:
                embedding_params += param.numel()
    
    # For standard BERT model structure
    elif hasattr(model, 'embeddings'):
        if hasattr(model.embeddings, 'word_embeddings'):
            embedding_params += model.embeddings.word_embeddings.weight.numel()
        # Check for other embedding types
        for name, param in model.embeddings.named_parameters():
            if name != 'word_embeddings.weight' and 'embed' in name:
                embedding_params += param.numel()
    
    # Calculate model-only parameters (excluding embeddings)
    model_only_params = total_params - embedding_params
    
    return {
        'total': total_params,
        'embedding': embedding_params,
        'model_only': model_only_params
    }

def prepare_datasets_with_mapping(task_name, token_map, oov_lookup=None, tokenizer=None, batch_size=512):
    """
    Prepare datasets for training with reduced vocabulary with GPU acceleration.
    Works for all pruning methods.
    
    Args:
        task_name: Name of the GLUE task
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping from OOV token ID to cluster representative ID (for hybrid methods)
        tokenizer: Tokenizer for the original model
        batch_size: Batch size for preprocessing
        
    Returns:
        train_dataset: Training dataset with remapped token IDs
        eval_dataset: Evaluation dataset with remapped token IDs
        test_dataset: Test dataset with remapped token IDs (if available)
    """
    # Set multiprocessing method for tokenizer
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method already set
        pass
    
    # Load dataset
    dataset_name = "mnli" if task_name.startswith("mnli") else task_name
    raw_datasets = load_dataset("glue", dataset_name)
    
    # Handle 'mnli' task name specially for task metadata
    task_key = "mnli-matched" if task_name == "mnli" else task_name
    task_meta = glue_tasks[task_key]
    train_ds_name = task_meta["dataset_names"]["train"]
    valid_ds_name = task_meta["dataset_names"]["valid"]

    # Print available splits for debugging
    logger.info(f"Available splits: {list(raw_datasets.keys())}")
    logger.info(f"Using train: {train_ds_name}, validation: {valid_ds_name}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get the UNK token ID in the new mapping
    unk_token_id = 0
    
    # Create tensor mapping for both direct mapping and OOV lookup
    # Create a tensor mapping from old to new token IDs
    # Add +1 to max_token_id to handle the case where the max token ID is exactly at the boundary
    max_token_id = max(tokenizer.vocab_size, max(token_map.keys()) + 2)  # Add extra padding to avoid index errors
    id_map_tensor = torch.full((max_token_id,), unk_token_id, dtype=torch.long, device=device)
    
    # First fill with direct mappings
    for old_id, new_id in token_map.items():
        id_map_tensor[old_id] = new_id
    
    # Then add OOV lookup mappings if provided
    if oov_lookup:
        for old_id, new_id in oov_lookup.items():
            id_map_tensor[old_id] = new_id
    
    sentence1_key, sentence2_key = task_to_keys[dataset_name]
    
    # Process a split using efficient GPU batching
    def transform_split_gpu(dataset_split):
        # For large datasets, use larger batch sizes to maximize GPU utilization
        if torch.cuda.is_available():
            # Calculate batch size based on GPU memory
            mem_info = torch.cuda.mem_get_info()
            free_mem = mem_info[0] / 1024**2  # Free memory in MB
            total_mem = mem_info[1] / 1024**2  # Total memory in MB
            
            # Adjust batch size based on GPU memory
            if "mnli" in task_name:
                if total_mem > 40000:  # For GPUs with more than 40GB VRAM
                    process_batch_size = 4096
                elif total_mem > 24000:  # For GPUs with 24-40GB VRAM
                    process_batch_size = 2048
                else:  # For smaller GPUs
                    process_batch_size = 1024
            else:
                # For smaller datasets, we can use more aggressive settings
                if total_mem > 40000:  # For large GPUs
                    process_batch_size = 8192
                elif total_mem > 24000:  # For medium GPUs
                    process_batch_size = 4096
                else:  # For smaller GPUs
                    process_batch_size = 2048
            
            logger.info(f"Using batch size {process_batch_size} for dataset processing based on GPU memory: {total_mem:.2f}MB")
        else:
            # CPU fallback
            process_batch_size = 256
        
        all_items = []
        
        # Process in batches
        for i in tqdm(range(0, len(dataset_split[sentence1_key]), process_batch_size),
                     desc="Processing examples"):
            # Get batch texts
            batch_texts1 = dataset_split[sentence1_key][i:i+process_batch_size]
            batch_texts2 = None if sentence2_key is None else dataset_split[sentence2_key][i:i+process_batch_size]
            
            # Try to get labels, but handle cases where labels might be missing or -1
            try:
                batch_labels = dataset_split['label'][i:i+process_batch_size]
            except (KeyError, IndexError):
                # If labels are not available, use dummy values (0) 
                # These won't be used for evaluation on test sets
                batch_labels = [0] * len(batch_texts1)
            
            # Tokenize with optimal settings
            if batch_texts2 is None:
                encodings = tokenizer(
                    batch_texts1, 
                    padding='max_length', 
                    truncation=True,
                    max_length=128,  # Shorter max_length for faster processing
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
            else:
                encodings = tokenizer(
                    batch_texts1, 
                    batch_texts2, 
                    padding='max_length', 
                    truncation=True,
                    max_length=128,  # Shorter max_length for faster processing
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
            
            # Move to GPU for faster processing
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            
            # Ensure no out-of-bounds indices
            input_ids = torch.clamp(input_ids, 0, max_token_id - 1)
            
            # Map token IDs using the mapping tensor (on GPU)
            new_input_ids = id_map_tensor[input_ids]
            
            # Create batch items
            for j in range(len(new_input_ids)):
                item = {
                    'input_ids': new_input_ids[j].cpu().tolist(),
                    'attention_mask': attention_mask[j].cpu().tolist(),
                    'labels': batch_labels[j]
                }
                all_items.append(item)
            
            # Clear GPU cache every few batches
            if i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return Dataset.from_list(all_items)
    
    # Transform train and validation sets
    logger.info(f"Processing train split ({len(raw_datasets[train_ds_name])} examples)...")
    train_dataset = transform_split_gpu(raw_datasets[train_ds_name])
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"Processing validation split ({len(raw_datasets[valid_ds_name])} examples)...")
    eval_dataset = transform_split_gpu(raw_datasets[valid_ds_name])
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get test dataset name if available
    test_ds_name = task_meta["dataset_names"].get("test")
    
    # Process test set if available
    if test_ds_name and test_ds_name in raw_datasets:
        logger.info(f"Processing test split ({len(raw_datasets[test_ds_name])} examples)...")
        test_dataset = transform_split_gpu(raw_datasets[test_ds_name])
    else:
        test_dataset = None
    
    return train_dataset, eval_dataset, test_dataset 

def setup_clustering_based_model(task_name, model_name, prune_percent=0, clustering_method="agglomerative"):
    """
    Set up a model with reduced vocabulary based on clustering-based pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune through clustering
        clustering_method: Method to use for clustering
        
    Returns:
        model: Model with reduced vocabulary
        token_map: Mapping from original token IDs to new IDs
    """
    logger.info(f"Setting up clustering-based model for {task_name} with {prune_percent}% pruning")
    
    # Get task metadata - handle 'mnli' task name specially
    task_key = "mnli-matched" if task_name == "mnli" else task_name
    task_meta = glue_tasks[task_key]
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    task_vocab = get_dataset_vocabulary(vocab_name, train_only=True)
    
    # Load model for clustering
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Apply clustering-based pruning if requested
    if prune_percent > 0:
        task_vocab = cluster_embeddings(task_vocab, model, prune_percent, clustering_method)
    
    # Create reduced embeddings
    token_map, reduced_embeddings = create_reduced_embeddings(task_vocab, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, None  # No OOV lookup for clustering method

def setup_frequency_based_model(task_name, model_name, prune_percent=0):
    """
    Set up a model with reduced vocabulary based on frequency-based pruning.
    This is equivalent to the clustering method but using frequency-based token selection.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune based on frequency
        
    Returns:
        model: Model with reduced vocabulary
        token_map: Mapping from original token IDs to new IDs
    """
    logger.info(f"Setting up frequency-based model for {task_name} with {prune_percent}% pruning")
    
    # Get task metadata - handle 'mnli' task name specially
    task_key = "mnli-matched" if task_name == "mnli" else task_name
    task_meta = glue_tasks[task_key]
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with counts
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, all_token_ids = get_dataset_tokens_with_counts(vocab_name, train_only=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Apply frequency-based pruning if requested
    if prune_percent > 0:
        tokens_to_keep, _ = frequency_based_pruning(token_counter, prune_percent)
    else:
        tokens_to_keep = sorted(list(all_token_ids))
    
    # Create reduced embeddings
    token_map, reduced_embeddings = create_reduced_embeddings(tokens_to_keep, model)
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    return model, token_map, None  # No OOV lookup for frequency method

def setup_hybrid_model(task_name, model_name, prune_percent=20, num_clusters=50):
    """
    Set up a model with hybrid vocabulary pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune based on frequency
        num_clusters: Number of clusters for OOV token mapping
        
    Returns:
        model: Model with hybrid vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping from OOV token ID to cluster representative ID
    """
    logger.info(f"Setting up hybrid model for {task_name} with {prune_percent}% pruning and {num_clusters} OOV clusters")
    
    # Get task metadata - handle 'mnli' task name specially
    task_key = "mnli-matched" if task_name == "mnli" else task_name
    task_meta = glue_tasks[task_key]
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with token counts
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, all_token_ids = get_dataset_tokens_with_counts(vocab_name, train_only=True)
    
    # Load model for embedding extraction
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Step 1: Frequency-based pruning
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

def setup_importance_based_model(task_name, model_name, prune_percent=20, num_clusters=50, importance_type=3):
    """
    Set up a model with word importance based vocabulary pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune based on importance
        num_clusters: Number of clusters for OOV token mapping
        importance_type: Word importance setting (0=frequency only, 1-3=TF-IDF variants)
        
    Returns:
        model: Model with importance-based vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping from OOV token ID to cluster representative ID
    """
    logger.info(f"Setting up importance-based model for {task_name} with {prune_percent}% pruning, {num_clusters} OOV clusters, importance_type={importance_type}")
    
    # Get task metadata - handle 'mnli' task name specially
    task_key = "mnli-matched" if task_name == "mnli" else task_name
    task_meta = glue_tasks[task_key]
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

def setup_training(model, train_dataset, eval_dataset, task_name, args):
    """
    Set up training and evaluation for a reduced vocabulary model.
    
    Args:
        model: Model with reduced vocabulary
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        task_name: Name of the GLUE task
        args: Command-line arguments with training hyperparameters
        
    Returns:
        trainer: Configured Trainer object
        metrics_callback: Callback for tracking metrics
    """
    # Training hyperparameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available(),  # Use bfloat16 if available
        bf16_full_eval=torch.cuda.is_available(),
        push_to_hub=False,
        seed=args.seed,
        torch_compile=False,  # Disable torch compilation to avoid CUDA errors
    )
    
    # Create data collator based on pruning method
    if args.pruning_method in ["hybrid", "importance"]:
        data_collator = HybridCollator(pad_token_id=0, unk_token_id=0)
    else:
        data_collator = ReducedVocabDataCollator(pad_token_id=0)
    
    # Setup metrics callback
    metrics_callback = MetricsCallback()
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, task_name=task_name),
    )
    
    # Add callback
    trainer.add_callback(metrics_callback)
    
    return trainer, metrics_callback

def run_pipeline(args):
    """
    Execute the full training pipeline with vocabulary pruning.
    
    Args:
        args: Command-line arguments
        
    Returns:
        results_df: DataFrame with training results
        model: Trained model
    """
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file
    log_filename = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}.log"
    if args.pruning_method == "clustering":
        log_filename = f"{args.output_dir}/{args.task}_clustering_prune{args.prune_percent}_{args.clustering_method}.log"
    elif args.pruning_method == "hybrid":
        log_filename = f"{args.output_dir}/{args.task}_hybrid_prune{args.prune_percent}_clusters{args.num_clusters}.log"
    elif args.pruning_method == "importance":
        log_filename = f"{args.output_dir}/{args.task}_importance_prune{args.prune_percent}_clusters{args.num_clusters}_type{args.importance_type}.log"
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    
    # Initialize tokenizer for the original model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Setup model with appropriate pruning method
    if args.pruning_method == "clustering":
        model, token_map, oov_lookup = setup_clustering_based_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            clustering_method=args.clustering_method
        )
    elif args.pruning_method == "frequency":
        model, token_map, oov_lookup = setup_frequency_based_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent
        )
    elif args.pruning_method == "hybrid":
        model, token_map, oov_lookup = setup_hybrid_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            num_clusters=args.num_clusters
        )
    elif args.pruning_method == "importance":
        model, token_map, oov_lookup = setup_importance_based_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            num_clusters=args.num_clusters,
            importance_type=args.importance_type
        )
    
    # Prepare datasets - either with custom splits or original GLUE splits
    if args.use_custom_splits:
        logger.info("Using custom train/validation/test splits")
        try:
            # Import custom splitting utilities
            from split_utils import prepare_custom_split_datasets
            
            # Create custom splits
            train_dataset, eval_dataset, test_dataset = prepare_custom_split_datasets(
                task_name=args.task,
                tokenizer=tokenizer,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                test_ratio=args.test_ratio,
                max_length=128,  # Same as in original prepare_datasets_with_mapping
                random_seed=args.seed
            )
            
            # Apply token remapping to create datasets with reduced vocabulary
            logger.info("Applying vocabulary mapping to custom splits")
            
            # Function to remap tokens in a dataset
            def remap_tokens(dataset):
                # Create a mapping dictionary for quick lookup
                id_map_dict = token_map.copy()
                if oov_lookup:
                    id_map_dict.update(oov_lookup)
                
                def map_example(example):
                    # Map input_ids using our token mapping
                    new_input_ids = []
                    for token_id in example['input_ids']:
                        if token_id in id_map_dict:
                            new_input_ids.append(id_map_dict[token_id])
                        else:
                            # Use UNK token (0) for OOV tokens
                            new_input_ids.append(0)
                    
                    example['input_ids'] = new_input_ids
                    return example
                
                # Apply mapping to each example
                return dataset.map(map_example, desc="Remapping token IDs")
            
            # Apply remapping to all datasets
            train_dataset = remap_tokens(train_dataset)
            eval_dataset = remap_tokens(eval_dataset)
            test_dataset = remap_tokens(test_dataset)
            
            logger.info(f"Created custom splits with sizes: train={len(train_dataset)}, validation={len(eval_dataset)}, test={len(test_dataset)}")
            
        except ImportError as e:
            logger.error(f"Failed to import split_utils module: {e}")
            logger.warning("Falling back to original GLUE splits")
            
            # If custom splitting fails, fall back to original method
            train_dataset, eval_dataset, test_dataset = prepare_datasets_with_mapping(
                args.task, 
                token_map, 
                oov_lookup,
                tokenizer, 
                batch_size=args.batch_size
            )
    else:
        # Use original GLUE splits
        train_dataset, eval_dataset, test_dataset = prepare_datasets_with_mapping(
            args.task, 
            token_map, 
            oov_lookup,
            tokenizer, 
            batch_size=args.batch_size
        )
    
    # Setup trainer
    trainer, metrics_callback = setup_training(
        model, 
        train_dataset, 
        eval_dataset, 
        args.task, 
        args
    )
    
    # Log initial vocabulary statistics
    logger.info(f"\n=== Vocabulary Statistics ===")
    logger.info(f"Original vocabulary size: {len(tokenizer.get_vocab())}")
    logger.info(f"Kept tokens: {len(token_map)}")
    if oov_lookup:
        logger.info(f"OOV clusters: {len(set(oov_lookup.values()))}")
    logger.info(f"Vocabulary reduction: {(1 - len(token_map)/len(tokenizer.get_vocab()))*100:.2f}%")
    
    # Log whether we're using custom splits
    if args.use_custom_splits:
        logger.info(f"Using custom splits with ratios: train={args.train_ratio}, val={args.validation_ratio}, test={args.test_ratio}")
    
    # Train and evaluate
    logger.info(f"Starting training for {args.epochs} epochs")
    try:
        trainer.train()
        
        # Evaluate the model
        logger.info("Running final evaluation")
        eval_results = trainer.evaluate()
        for metric_name, value in eval_results.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Get training history as DataFrame with error handling
        try:
            # Check if we have training history
            if metrics_callback.training_history["train"] and metrics_callback.training_history["eval"]:
                # Handle potential arrays with different lengths
                train_data = metrics_callback.training_history["train"]
                
                # Find min length of all arrays
                min_len = min(len(arr) for arr in train_data.values())
                
                # Truncate all arrays to the same length
                train_data_truncated = {k: v[:min_len] for k, v in train_data.items()}
                
                # Create DataFrames
                train_history_df = pd.DataFrame(train_data_truncated)
                train_history_df = train_history_df.add_prefix("train_")
                
                # Do the same for eval metrics
                eval_data = metrics_callback.training_history["eval"]
                min_len_eval = min(len(arr) for arr in eval_data.values() if len(arr) > 0)
                eval_data_truncated = {k: v[:min_len_eval] for k, v in eval_data.items() if len(v) > 0}
                
                if eval_data_truncated:
                    eval_history_df = pd.DataFrame(eval_data_truncated)
                    # Combine only if there's data to combine
                    if not eval_history_df.empty and not train_history_df.empty:
                        results_df = pd.concat([train_history_df, eval_history_df], axis=1)
                    else:
                        results_df = train_history_df if not train_history_df.empty else eval_history_df
                else:
                    results_df = train_history_df
                
                # Save results
                results_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_results.csv"
                results_df.to_csv(results_file)
                logger.info(f"Saved training results to {results_file}")
            else:
                logger.warning("No training history available, skipping results DataFrame creation")
                results_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Error creating results DataFrame: {e}")
            logger.warning("Continuing without saving training history")
            results_df = pd.DataFrame()
        
        # Analyze model size
        model_params = count_model_parameters(trainer.model)
        
        # Load original model for comparison
        logger.info("\nAnalyzing original model for comparison")
        original_model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        original_params = count_model_parameters(original_model)
        
        # Calculate reduction percentages
        logger.info("\n=== Parameter Reduction Statistics ===")
        for param_type in ['total', 'embedding', 'model_only']:
            reduction = (original_params[param_type] - model_params[param_type]) / original_params[param_type] * 100
            logger.info(f"{param_type.title()} parameter reduction: {reduction:.2f}%")
        
        # Log summary of results
        logger.info("\n=== Summary Results ===")
        logger.info(f"Task: {args.task}")
        logger.info(f"Pruning method: {args.pruning_method}")
        logger.info(f"Prune percent: {args.prune_percent}%")
        if args.pruning_method in ["hybrid", "importance"]:
            logger.info(f"Num OOV clusters: {args.num_clusters}")
        if args.pruning_method == "importance":
            logger.info(f"Importance type: {args.importance_type}")
        logger.info(f"Vocabulary reduction: {(1 - len(token_map)/len(tokenizer.get_vocab()))*100:.2f}%")
        
        # Extract final metrics
        final_metrics = {}
        for metric_name, value in eval_results.items():
            if metric_name.startswith("eval_"):
                metric_short_name = metric_name[5:]  # Remove 'eval_' prefix
                final_metrics[metric_short_name] = value
        
        # Log performance metrics
        for metric_name, value in final_metrics.items():
            logger.info(f"Final {metric_name}: {value:.4f}")
        
        # After training, generate predictions on test set if available
        if test_dataset is not None:
            logger.info("\n=== Test Set Prediction Generation ===")
            try:
                # Create a custom forward function for prediction without loss calculation
                def custom_forward(model, input_ids, attention_mask, **kwargs):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    return {"logits": outputs.logits}
                    
                # Generate predictions without computing loss
                logger.info("Generating predictions on test set")
                with torch.no_grad():
                    device = next(model.parameters()).device
                    predictions = []
                    
                    # Process in batches
                    dataloader = torch.utils.data.DataLoader(
                        test_dataset, batch_size=args.batch_size,
                        collate_fn=trainer.data_collator
                    )
                    
                    for batch in tqdm(dataloader, desc="Predicting"):
                        # Filter out non-tensor values and labels
                        batch = {k: v.to(device) for k, v in batch.items() 
                                if k != 'labels' and k != 'prediction_only' and hasattr(v, 'to')}
                        with torch.no_grad():
                            outputs = model(**batch)
                        logits = outputs.logits
                        predictions.append(logits.cpu().numpy())
                    
                    # Concatenate predictions
                    all_predictions = np.vstack(predictions)
                    
                    # For classification tasks, get the class with highest probability
                    if len(all_predictions.shape) > 1 and all_predictions.shape[1] > 1:
                        predictions = np.argmax(all_predictions, axis=1)
                    else:
                        # For regression tasks
                        predictions = all_predictions.squeeze()
                
                # Save predictions for submission
                output_test_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_predictions.txt"
                np.savetxt(output_test_file, predictions, fmt='%d' if len(predictions.shape) == 1 else '%.6f')
                logger.info(f"Saved test predictions to {output_test_file}")
                
            except Exception as e:
                logger.error(f"Error generating test predictions: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        return results_df, trainer.model
    
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error occurred: {e}")
            logger.info("Attempting to free memory and continue with CPU...")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move model to CPU for evaluation
            model = model.cpu()
            
            # Create empty results dataframe
            results_df = pd.DataFrame()
            
            return results_df, model
        else:
            raise

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Run pipeline
    try:
        results_df, model = run_pipeline(args)
        logger.info("Pipeline completed successfully")
        return results_df, model
    except Exception as e:
        logger.exception(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 