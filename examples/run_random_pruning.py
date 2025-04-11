#!/usr/bin/env python
# coding: utf-8

"""
Random-based Vocabulary Pruning Example Script

This script demonstrates how to use the random-based vocabulary pruning technique
on GLUE benchmark tasks. This serves as a baseline approach where tokens are pruned randomly
without consideration for importance.

Example usage:
    python run_random_pruning.py --task mrpc --prune_percent 20
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

# Default settings for each task (customize as needed)
TASK_DEFAULTS = {
    "cola": {"batch_size": BATCH_SIZE, "learning_rate": 8e-5, "epochs": 5, "prune_percent": PRUNE_PERCENT},
    "mnli": {"batch_size": BATCH_SIZE, "learning_rate": 5e-5, "epochs": 1, "prune_percent": PRUNE_PERCENT},
    "mrpc": {"batch_size": BATCH_SIZE, "learning_rate": 5e-5, "epochs": 10, "prune_percent": PRUNE_PERCENT},
    "qnli": {"batch_size": BATCH_SIZE, "learning_rate": 8e-5, "epochs": 2, "prune_percent": PRUNE_PERCENT},
    "qqp": {"batch_size": BATCH_SIZE, "learning_rate": 5e-5, "epochs": 10, "prune_percent": PRUNE_PERCENT},
    "rte": {"batch_size": BATCH_SIZE, "learning_rate": 5e-5, "epochs": 3, "prune_percent": PRUNE_PERCENT},
    "sst2": {"batch_size": BATCH_SIZE, "learning_rate": 8e-5, "epochs": 2, "prune_percent": PRUNE_PERCENT},
    "stsb": {"batch_size": BATCH_SIZE, "learning_rate": 8e-5, "epochs": 10, "prune_percent": PRUNE_PERCENT},
    "wnli": {"batch_size": BATCH_SIZE, "learning_rate": 5e-5, "epochs": 3, "prune_percent": PRUNE_PERCENT}
}

def parse_args():
    """Parse command line arguments for Random pruning."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models with Random-based vocabulary pruning"
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
    """Main function to run Random-based pruning."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Apply task-specific defaults unless overridden
    task_name = args.task
    batch_size = args.batch_size if args.batch_size is not None else TASK_DEFAULTS[task_name]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else TASK_DEFAULTS[task_name]["learning_rate"]
    epochs = args.epochs if args.epochs is not None else TASK_DEFAULTS[task_name]["epochs"]
    prune_percent = args.prune_percent if args.prune_percent is not None else TASK_DEFAULTS[task_name]["prune_percent"]
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../results/{task_name}_random_prune{prune_percent}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up configuration for the run
    config = argparse.Namespace(
        task=task_name,
        model_name=args.model_name,
        pruning_method="random",  # Specify pruning method as random
        prune_percent=prune_percent,
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
    logger.info(f"Running Random-based pruning on {task_name} with {prune_percent}% pruning")
    logger.info(f"Output directory: {output_dir}")
    
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")
    
    # Run the pruning pipeline
    results_df, model = run_pipeline(config)
    
    logger.info(f"Random-based pruning completed successfully")
    
    return results_df, model

if __name__ == "__main__":
    main() 