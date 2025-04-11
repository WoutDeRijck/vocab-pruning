#!/usr/bin/env python
# coding: utf-8

"""
Unified Vocabulary Pruning Script - Main Entry Point

This script serves as the main entry point for the vocabulary pruning pipeline.
It parses command line arguments and executes the appropriate pruning method.
"""

import os
import argparse
import logging
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import local modules
from utils import setup_logging, set_seed, get_task_metadata
from metrics import count_model_parameters
from data import prepare_datasets_with_mapping
from training import setup_training, save_training_results, generate_test_predictions

# Import pruning methods
from pruning.clustering import setup_clustering_based_model
from pruning.frequency import setup_frequency_based_model
from pruning.hybrid import setup_hybrid_model
from pruning.importance import setup_importance_based_model

# Configure logging
logger = logging.getLogger(__name__)

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
        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
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

def run_pipeline(args):
    """
    Execute the full training pipeline with vocabulary pruning.
    
    Args:
        args: Command-line arguments
        
    Returns:
        model: Trained model
    """
    # Set up logging
    setup_logging(args)
    
    # Set random seeds
    set_seed(args.seed)
    
    # Initialize tokenizer for the original model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Setup model with appropriate pruning method
    logger.info(f"Setting up model with {args.pruning_method} pruning method")
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
    
    # Prepare datasets with token mapping
    logger.info("Preparing datasets with token mapping")
    train_dataset, eval_dataset, test_dataset = prepare_datasets_with_mapping(
        args.task, 
        token_map, 
        oov_lookup,
        tokenizer, 
        batch_size=args.batch_size
    )
    
    # Setup trainer
    logger.info("Setting up training")
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
    
    # Train and evaluate
    logger.info(f"Starting training for {args.epochs} epochs")
    try:
        # Run training
        trainer.train()
        
        # Evaluate the model
        logger.info("Running final evaluation")
        eval_results = trainer.evaluate()
        for metric_name, value in eval_results.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Save training results
        save_training_results(metrics_callback, args, args.task, args.output_dir)
        
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
        
        # Generate test predictions if available
        if test_dataset is not None:
            generate_test_predictions(trainer.model, test_dataset, trainer.data_collator, args)
        
        return trainer.model
    
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error occurred: {e}")
            logger.info("Attempting to free memory and continue with CPU...")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move model to CPU for evaluation
            model = model.cpu()
            
            return model
        else:
            logger.error(f"Error during training: {e}")
            logger.error(traceback.format_exc())
            raise

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        model = run_pipeline(args)
        logger.info("Pipeline completed successfully")
        return model
    except Exception as e:
        logger.exception(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 