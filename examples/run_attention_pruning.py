#!/usr/bin/env python
# coding: utf-8

"""
Attention-based Vocabulary Pruning for All GLUE Tasks

This script runs attention-based vocabulary pruning on all GLUE benchmark tasks.
It can be executed with a single command to process all tasks or a specific task.

Example usage:
    # Run a specific task
    python run_attention_pruning_for_all_tasks.py --task mrpc --prune_percent 20 --attention_model_path path/to/finetuned/model
    
    # Run all tasks
    python run_attention_pruning_for_all_tasks.py --run_all --prune_percent 20
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

# Default settings for each task (customize as needed)
# You can fill in the model_name fields with paths to task-specific fine-tuned models
TASK_DEFAULTS = {
    "cola": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 5,
        "model_name": ""  # Path to fine-tuned model for CoLA
    },
    "mnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 1,
        "model_name": ""  # Path to fine-tuned model for MNLI
    },
    "mrpc": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 10,
        "model_name": ""  # Path to fine-tuned model for MRPC
    },
    "qnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 2,
        "model_name": ""  # Path to fine-tuned model for QNLI
    },
    "qqp": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 10,
        "model_name": ""  # Path to fine-tuned model for QQP
    },
    "rte": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 1e-4, 
        "epochs": 3,
        "model_name": ""  # Path to fine-tuned model for RTE
    },
    "sst2": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 2,
        "model_name": ""  # Path to fine-tuned model for SST-2
    },
    "stsb": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 1e-4, 
        "epochs": 10,
        "model_name": ""  # Path to fine-tuned model for STS-B
    },
    "wnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 1e-4, 
        "epochs": 3,
        "model_name": ""  # Path to fine-tuned model for WNLI
    }
}

def parse_args():
    """Parse command line arguments for Attention-based pruning."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models with Attention-based vocabulary pruning for all GLUE tasks"
    )
    
    # Task arguments
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task", 
        type=str,
        choices=GLUE_TASKS,
        help="Specific GLUE task to run"
    )
    task_group.add_argument(
        "--run_all",
        action="store_true",
        help="Run all GLUE tasks sequentially"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name or path (used if task-specific model not provided)"
    )
    
    parser.add_argument(
        "--attention_model_path", 
        type=str, 
        default=None,
        help="Path to a fine-tuned model to use for attention-based importance calculation"
    )
    
    # Pruning arguments
    parser.add_argument(
        "--prune_percent", 
        type=float, 
        default=20,
        help="Percentage of vocabulary to prune"
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
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Base directory for results"
    )
    
    return parser.parse_args()

def run_task(task_name, args):
    """Run attention-based pruning for a specific task."""
    logger.info(f"{'='*50}")
    logger.info(f"Running attention-based pruning for {task_name}")
    logger.info(f"{'='*50}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Apply task-specific defaults unless overridden
    batch_size = args.batch_size if args.batch_size is not None else TASK_DEFAULTS[task_name]["batch_size"]
    learning_rate = args.learning_rate if args.learning_rate is not None else TASK_DEFAULTS[task_name]["learning_rate"]
    epochs = args.epochs if args.epochs is not None else TASK_DEFAULTS[task_name]["epochs"]
    
    # Use task-specific model if available, otherwise fall back to command line argument
    model_name = TASK_DEFAULTS[task_name]["model_name"] if TASK_DEFAULTS[task_name]["model_name"] else args.model_name
    
    # Determine attention model - if not explicitly provided, use base model
    attention_model = args.attention_model_path
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = "_finetuned" if attention_model else ""
    output_dir = f"{args.results_dir}/{task_name}_attention_prune{args.prune_percent}{model_suffix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up configuration for the run
    config = argparse.Namespace(
        task=task_name,
        model_name=model_name,
        pruning_method="attention",  # Specify pruning method as attention
        prune_percent=args.prune_percent,
        attention_model=attention_model,  # Pass the fine-tuned model path
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
    logger.info(f"Running with configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    
    # Run the pruning pipeline
    try:
        results_df, model = run_pipeline(config)
        logger.info(f"Attention-based pruning for {task_name} completed successfully")
        return results_df, model
    except Exception as e:
        logger.error(f"Error processing {task_name}: {e}")
        return None, None

def main():
    """Main function to run Attention-based pruning for all tasks."""
    args = parse_args()
    
    # Determine which tasks to run
    tasks_to_run = GLUE_TASKS if args.run_all else [args.task]
    
    results = {}
    
    # Run each task
    for task in tasks_to_run:
        results_df, model = run_task(task, args)
        results[task] = (results_df, model)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Attention-based pruning summary:")
    for task in tasks_to_run:
        status = "✅ Completed" if results[task][0] is not None else "❌ Failed"
        logger.info(f"{task}: {status}")
    logger.info("="*80)
    
    return results

if __name__ == "__main__":
    main() 