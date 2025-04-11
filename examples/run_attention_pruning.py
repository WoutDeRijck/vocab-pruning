#!/usr/bin/env python
# coding: utf-8

"""
Attention-based Vocabulary Pruning Example Script

This script demonstrates how to use the attention-based vocabulary pruning technique
on GLUE benchmark tasks. This approach uses attention patterns from a fine-tuned model
to determine token importance in context.

Example usage:
    python run_attention_pruning.py --task mrpc --prune_percent 20 --finetuned_model_path path/to/finetuned/model
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
# Fill in the finetuned_model field with paths to task-specific fine-tuned models
TASK_DEFAULTS = {
    "cola": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 5,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for CoLA
    },
    "mnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 1,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for MNLI
    },
    "mrpc": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for MRPC
    },
    "qnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 2,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for QNLI
    },
    "qqp": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for QQP
    },
    "rte": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 3,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for RTE
    },
    "sst2": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 2,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for SST-2
    },
    "stsb": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for STS-B
    },
    "wnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 3,
        "prune_percent": PRUNE_PERCENT,
        "finetuned_model": ""  # Path to fine-tuned model for WNLI
    }
}

def parse_args():
    """Parse command line arguments for Attention-based pruning."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models with Attention-based vocabulary pruning"
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
        help="Base model name or path (only used if no fine-tuned model is provided)"
    )
    
    parser.add_argument(
        "--finetuned_model_path", 
        type=str, 
        default=None,
        help="Path to a fine-tuned model to use for both embedding extraction and attention-based importance calculation"
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
    """Main function to run Attention-based pruning."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Apply task-specific defaults unless overridden
    task_name = args.task
    batch_size = args.batch_size if args.batch_size is not None else TASK_DEFAULTS[task_name]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else TASK_DEFAULTS[task_name]["learning_rate"]
    epochs = args.epochs if args.epochs is not None else TASK_DEFAULTS[task_name]["epochs"]
    prune_percent = args.prune_percent if args.prune_percent is not None else TASK_DEFAULTS[task_name]["prune_percent"]
    
    # Use task-specific fine-tuned model if available, otherwise use command line argument
    finetuned_model = args.finetuned_model_path
    if not finetuned_model and TASK_DEFAULTS[task_name]["finetuned_model"]:
        finetuned_model = TASK_DEFAULTS[task_name]["finetuned_model"]
    
    # Determine which model to use for embeddings and attention calculation
    model_name = finetuned_model if finetuned_model else args.model_name
    attention_model = finetuned_model  # Always use the same model for attention
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = "_finetuned" if finetuned_model else ""
    output_dir = f"../results/{task_name}_attention_prune{prune_percent}{model_suffix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up configuration for the run
    config = argparse.Namespace(
        task=task_name,
        model_name=model_name,
        pruning_method="attention",  # Specify pruning method as attention
        prune_percent=prune_percent,
        attention_model=attention_model,  # Pass the fine-tuned model path to attention calculation
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
    logger.info(f"Running Attention-based pruning on {task_name} with {prune_percent}% pruning")
    if finetuned_model:
        logger.info(f"Using fine-tuned model: {finetuned_model}")
    logger.info(f"Output directory: {output_dir}")
    
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")
    
    # Run the pruning pipeline
    results_df, model = run_pipeline(config)
    
    logger.info(f"Attention-based pruning completed successfully")
    
    return results_df, model

if __name__ == "__main__":
    main() 