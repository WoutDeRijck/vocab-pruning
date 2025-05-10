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

from .base import get_dataset_tokens_with_counts, create_hybrid_embeddings, TokenMappingWrapper
from .frequency import frequency_based_pruning
from utils import get_task_metadata

# Configure logging
logger = logging.getLogger(__name__)

def cluster_removed_tokens(tokens_to_remove, tokens_to_keep, model, num_clusters=50):
    """
    Cluster removed tokens to create mapping for OOV tokens.
    Used by frequency_oov pruning methods.
    
    Args:
        tokens_to_remove: List of token IDs that were removed
        tokens_to_keep: List of token IDs that are kept in the vocabulary
        model: Model whose embeddings will be used
        num_clusters: Number of clusters to create
        
    Returns:
        oov_lookup: Mapping from removed token ID to nearest cluster center token ID
        updated_tokens_to_keep: Updated list of tokens to keep (may include cluster centers)
    """
    if not tokens_to_remove:
        logger.info("No tokens to cluster (all tokens kept)")
        return {}, tokens_to_keep
        
    logger.info(f"Clustering {len(tokens_to_remove)} removed tokens into {num_clusters} clusters")
    
    # Get original embedding matrix
    original_embeddings = model.model.embeddings.tok_embeddings.weight.data
    
    # Get the maximum valid token ID (embedding size - 1)
    max_valid_token_id = original_embeddings.size(0) - 1
    
    # Filter out any invalid token IDs from tokens_to_remove and tokens_to_keep
    valid_tokens_to_remove = [t for t in tokens_to_remove if t <= max_valid_token_id]
    valid_tokens_to_keep = [t for t in tokens_to_keep if t <= max_valid_token_id]
    
    if len(valid_tokens_to_remove) < len(tokens_to_remove):
        logger.warning(f"Filtered out {len(tokens_to_remove) - len(valid_tokens_to_remove)} invalid token IDs from tokens_to_remove")
    
    if len(valid_tokens_to_keep) < len(tokens_to_keep):
        logger.warning(f"Filtered out {len(tokens_to_keep) - len(valid_tokens_to_keep)} invalid token IDs from tokens_to_keep")
    
    # Use the valid tokens
    tokens_to_remove = valid_tokens_to_remove
    tokens_to_keep = valid_tokens_to_keep
    
    # If after filtering, we have no tokens to remove, return
    if not tokens_to_remove:
        logger.info("No valid tokens to cluster after filtering")
        return {}, tokens_to_keep
    
    # Get embeddings for removed tokens
    removed_embeddings = original_embeddings[tokens_to_remove].cpu().numpy()
    
    # Adjust number of clusters if we have fewer tokens than requested clusters
    actual_num_clusters = min(num_clusters, len(tokens_to_remove))
    
    # Perform KMeans clustering (faster than agglomerative for large number of tokens)
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(removed_embeddings)
    
    # Initialize mapping from removed token to nearest cluster center token
    oov_lookup = {}
    cluster_centers = []
    # Create a copy of tokens_to_keep that we can potentially update
    updated_tokens_to_keep = tokens_to_keep.copy()
    
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
            
            # Add the center token to tokens_to_keep if it's not already there
            if center_token_id not in updated_tokens_to_keep:
                updated_tokens_to_keep.append(center_token_id)
                logger.info(f"Added cluster center token {center_token_id} to tokens_to_keep")
            
            # Map all tokens in this cluster to the center token
            for idx in cluster_indices:
                oov_lookup[tokens_to_remove[idx]] = center_token_id
            
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
    
    return oov_lookup, updated_tokens_to_keep

def setup_frequency_oov_model(task_name, model_name, prune_percent=20, num_clusters=50, param_based=False):
    """
    Set up a model with frequency-based OOV vocabulary pruning.
    
    Args:
        task_name: Name of the GLUE task
        model_name: Base model to use
        prune_percent: Percentage of tokens or parameters to prune based on frequency
        num_clusters: Number of clusters for OOV token mapping
        param_based: If True, prune based on parameter percentage rather than token percentage
        
    Returns:
        model: Model with frequency OOV vocabulary
        token_map: Mapping from original token IDs to new IDs
        oov_lookup: Mapping from OOV token ID to cluster representative ID
    """
    pruning_type = "parameter-based" if param_based else "token-based"
    logger.info(f"Setting up {pruning_type} frequency-OOV model for {task_name} with {prune_percent}% pruning and {num_clusters} OOV clusters")
    
    # Load GLUE task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Get dataset vocabulary with token counts
    vocab_name = "mnli" if task_name.startswith("mnli") else task_name
    token_counter, all_token_ids = get_dataset_tokens_with_counts(vocab_name, train_only=True)
    
    # Load model for embedding extraction
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    
    # Get total model parameters and embedding dimension
    total_params = sum(p.numel() for p in model.parameters())
    embedding_dim = model.model.embeddings.tok_embeddings.embedding_dim
    total_vocab_size = model.model.embeddings.tok_embeddings.num_embeddings
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Token embeddings: {total_vocab_size:,} tokens with dimension {embedding_dim}")
    
    # Step 1: Frequency-based pruning
    tokens_to_keep, tokens_to_remove = frequency_based_pruning(
        token_counter,
        prune_percent,
        param_based=param_based,
        embedding_dim=embedding_dim,
        total_params=total_params,
        total_vocab_size=total_vocab_size
    )
    
    # Step 2: Cluster removed tokens
    oov_lookup, updated_tokens_to_keep = cluster_removed_tokens(tokens_to_remove, tokens_to_keep, model, num_clusters)
    
    # Step 3: Create hybrid embeddings
    logger.info(f"Creating hybrid embeddings for {len(updated_tokens_to_keep)} kept tokens and {len(oov_lookup)} OOV tokens")
    token_map, reduced_embeddings = create_hybrid_embeddings(updated_tokens_to_keep, oov_lookup, model)
    
    # Log embedding size before replacing
    logger.info(f"Original embedding size: {model.model.embeddings.tok_embeddings.weight.size()}")
    logger.info(f"Reduced embedding size: {reduced_embeddings.size()}")
    
    # Replace embedding layer with reduced version
    model.model.embeddings.tok_embeddings = nn.Embedding.from_pretrained(
        reduced_embeddings, freeze=False
    )
    
    # Verify the embedding size after replacement
    logger.info(f"New embedding size: {model.model.embeddings.tok_embeddings.weight.size()}")
    
    # Step 4: Wrap the model to handle token remapping during forward pass
    wrapped_model = TokenMappingWrapper(model, token_map, oov_lookup)
    
    return wrapped_model, token_map, oov_lookup 