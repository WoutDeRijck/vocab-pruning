"""
Clustering-based vocabulary pruning.

This module implements vocabulary pruning based on token embedding clustering.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from transformers import AutoTokenizer

from .base import get_dataset_vocabulary, create_reduced_embeddings

# Configure logging
logger = logging.getLogger(__name__)

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
        oov_lookup: None for clustering method
    """
    logger.info(f"Setting up clustering-based model for {task_name} with {prune_percent}% pruning")
    
    # Load GLUE task metadata
    from ..utils import get_task_metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    task_vocab = get_dataset_vocabulary(vocab_name, train_only=True)
    
    # Load model for clustering
    from transformers import AutoModelForSequenceClassification
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