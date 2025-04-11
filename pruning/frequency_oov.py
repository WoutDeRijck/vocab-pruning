"""
Frequency-based OOV vocabulary pruning.

This module implements frequency-based OOV pruning, which combines frequency-based pruning
with clustering of OOV tokens.
"""

import logging
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import get_dataset_tokens_with_counts, create_hybrid_embeddings
from .frequency import frequency_based_pruning
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def cluster_removed_tokens(tokens_to_remove, model, num_clusters=50):
    """
    Cluster removed tokens to create mapping for OOV tokens.
    Used by frequency_oov pruning methods.
    
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

def setup_frequency_oov_model(task_name, model_name, prune_percent=20, num_clusters=50):
    """
    Set up a model with frequency-based OOV vocabulary pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens to prune based on frequency
        num_clusters: Number of clusters for OOV token mapping
        
    Returns:
        model: Model with frequency OOV vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping from OOV token ID to cluster representative ID
    """
    logger.info(f"Setting up frequency-OOV model for {task_name} with {prune_percent}% pruning and {num_clusters} OOV clusters")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
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