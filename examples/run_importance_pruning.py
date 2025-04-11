#!/usr/bin/env python
# coding: utf-8

"""
Word Importance-based Vocabulary Pruning Example Script (Without OOV)

This script demonstrates how to use the word importance-based vocabulary pruning technique
without OOV clustering on GLUE benchmark tasks. Tokens are pruned based on TF-IDF importance,
with OOV tokens mapped to UNK (unlike importance_oov which maps them to cluster representatives).

Example usage:
    python run_importance_pruning.py --task mrpc --prune_percent 20 --importance_type 3
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append('..')

from main import run_pipeline
from utils import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define all GLUE tasks
GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

BATCH_SIZE = 64
PRUNE_PERCENT = 20
IMPORTANCE_TYPE = 3

# Default settings for each task (customize as needed)
TASK_DEFAULTS = {
    "cola": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 5,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "mnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 1,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "mrpc": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "qnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 2,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "qqp": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "rte": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 3,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "sst2": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 2,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "stsb": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    },
    "wnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 3,
        "prune_percent": PRUNE_PERCENT,
        "importance_type": IMPORTANCE_TYPE
    }
}

def parse_args():
    """Parse command line arguments for Word Importance pruning without OOV."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models with Word Importance-based vocabulary pruning (without OOV)"
    )
    
    # Task and model arguments
    parser.add_argument(
        "--task", 
        type=str, 
        default="mrpc", 
        choices=GLUE_TASKS,
        help="GLUE task name"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name or path"
    )
    
    # Pruning arguments
    parser.add_argument(
        "--prune_percent", 
        type=float, 
        default=None,
        help="Percentage of vocabulary to prune (overrides task defaults)"
    )
    
    parser.add_argument(
        "--importance_type", 
        type=int, 
        default=None,
        choices=[0, 1, 2, 3],
        help="Word importance calculation type: 0=off, 1=no norm, 2=L1 norm, 3=L2 norm (overrides task defaults)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs (overrides task defaults)"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None,
        help="Learning rate (overrides task defaults)"
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
        default=None,
        help="Training batch size (overrides task defaults)"
    )
    
    # Cross-validation arguments
    parser.add_argument(
        "--cross_validation", 
        action="store_true",
        help="Use cross-validation"
    )
    
    parser.add_argument(
        "--n_folds", 
        type=int, 
        default=5,
        help="Number of folds for cross-validation"
    )
    
    # Data split arguments
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.8,
        help="Ratio of data to use for training"
    )
    
    parser.add_argument(
        "--validation_ratio", 
        type=float, 
        default=0.1,
        help="Ratio of data to use for validation"
    )
    
    parser.add_argument(
        "--test_ratio", 
        type=float, 
        default=0.1,
        help="Ratio of data to use for testing"
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def main():
    """Main function to run Word Importance-based pruning without OOV."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Apply task-specific defaults unless overridden
    task_name = args.task
    batch_size = args.batch_size if args.batch_size is not None else TASK_DEFAULTS[task_name]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else TASK_DEFAULTS[task_name]["learning_rate"]
    epochs = args.epochs if args.epochs is not None else TASK_DEFAULTS[task_name]["epochs"]
    prune_percent = args.prune_percent if args.prune_percent is not None else TASK_DEFAULTS[task_name]["prune_percent"]
    importance_type = args.importance_type if args.importance_type is not None else TASK_DEFAULTS[task_name]["importance_type"]
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../results/{task_name}_importance_prune{prune_percent}_type{importance_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up configuration for the run
    config = argparse.Namespace(
        task=task_name,
        model_name=args.model_name,
        pruning_method="importance",  # Specify pruning method as importance (no OOV)
        prune_percent=prune_percent,
        importance_type=importance_type,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        batch_size=batch_size,
        cross_validation=args.cross_validation,
        n_folds=args.n_folds,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        output_dir=output_dir,
        seed=args.seed
    )
    
    # Log configuration
    logger.info(f"Running Word Importance-based pruning (without OOV) on {task_name} with {prune_percent}% pruning")
    logger.info(f"Importance type: {importance_type}")
    logger.info(f"Output directory: {output_dir}")
    
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")
    
    # Run the pruning pipeline
    results_df, model = run_pipeline(config)
    
    logger.info(f"Word Importance-based pruning (without OOV) completed successfully")
    
    return results_df, model

if __name__ == "__main__":
    main() 