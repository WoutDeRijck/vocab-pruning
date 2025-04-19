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
import subprocess
import re
import pandas as pd
from datetime import datetime
import glob

# Ensure NLTK resources are available
try:
    import nltk
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.warning(f"Could not download NLTK stopwords: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Get the absolute path to main.py
MAIN_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "main.py")

# Define all GLUE tasks
GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

BATCH_SIZE = 128
PRUNE_PERCENT = 20
IMPORTANCE_TYPE = 3

# Default settings for each task (customize as needed)
TASK_DEFAULTS = {
    "cola": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 8,
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
        "epochs": 15,
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
        "epochs": 10,
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
        "--tasks", 
        type=str, 
        nargs="+",
        default=list(TASK_DEFAULTS.keys()),
        help="GLUE tasks to run (default: all)"
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
    
    # Output directory
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./importance_pruning_output",
        help="Output directory for model checkpoints and logs"
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

def parse_log_file(log_file):
    """
    Parse a log file to extract key metrics.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
            
            # Extract validation metrics
            val_metrics = re.findall(r'eval_(\w+): ([0-9.]+)', log_content)
            for metric_name, value in val_metrics:
                metrics[f"val_{metric_name}"] = float(value)
            
            # Extract test metrics
            test_section = re.search(r'=== Test Set Evaluation Results ===\n(.*?)(?=\n\n|\Z)', 
                                    log_content, re.DOTALL)
            if test_section:
                test_metrics = re.findall(r'(\w+): ([0-9.]+)', test_section.group(1))
                for metric_name, value in test_metrics:
                    metrics[f"test_{metric_name}"] = float(value)
            
            # Extract parameter reduction statistics
            param_section = re.search(r'=== Parameter Reduction Statistics ===\n(.*?)(?=\n\n|\Z)', 
                                    log_content, re.DOTALL)
            if param_section:
                param_metrics = re.findall(r'(\w+) parameter reduction: ([0-9.]+)%', 
                                        param_section.group(1))
                for param_type, value in param_metrics:
                    metrics[f"{param_type.lower()}_param_reduction"] = float(value)
            
            # Extract vocabulary reduction
            vocab_match = re.search(r'Vocabulary reduction: ([0-9.]+)%', log_content)
            if vocab_match:
                metrics["vocab_reduction"] = float(vocab_match.group(1))
            
    except Exception as e:
        logger.error(f"Error parsing log file {log_file}: {e}")
    
    return metrics

def main():
    """Main function to run Word Importance-based pruning without OOV on multiple tasks."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    
    # Dictionary to store results for each task
    all_results = {}
    
    # Process each task
    for task_name in args.tasks:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running Word Importance-based pruning for task: {task_name}")
        logger.info(f"{'=' * 50}")
        
        # Apply task-specific defaults unless overridden
        batch_size = args.batch_size if args.batch_size is not None else TASK_DEFAULTS[task_name]["batch_size"]
        learning_rate = args.learning_rate if args.learning_rate is not None else TASK_DEFAULTS[task_name]["learning_rate"]
        epochs = args.epochs if args.epochs is not None else TASK_DEFAULTS[task_name]["epochs"]
        prune_percent = args.prune_percent if args.prune_percent is not None else TASK_DEFAULTS[task_name]["prune_percent"]
        importance_type = args.importance_type if args.importance_type is not None else TASK_DEFAULTS[task_name]["importance_type"]
        
        # Create task-specific output directory
        task_output_dir = os.path.join(args.output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Log task parameters
        logger.info(f"Parameters: batch_size={batch_size}, learning_rate={learning_rate}, "
                  f"epochs={epochs}, prune_percent={prune_percent}, "
                  f"importance_type={importance_type}")
        
        # Build command for the main script
        cmd = [
            "python", MAIN_SCRIPT_PATH,
            "--task", task_name,
            "--model_name", args.model_name,
            "--pruning_method", "importance",
            "--prune_percent", str(prune_percent),
            "--epochs", str(epochs),
            "--learning_rate", str(learning_rate),
            "--batch_size", str(batch_size),
            "--weight_decay", str(args.weight_decay),
            "--importance_type", str(importance_type),
            "--output_dir", task_output_dir,
            "--seed", str(args.seed),
        ]
        
        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully completed importance pruning for task: {task_name}")
            
            # Find the log file for this task
            log_file = os.path.join(
                task_output_dir, 
                f"{task_name}_importance_prune{int(prune_percent)}_type{importance_type}.log"
            )
            # Try alternative filename if not found
            if not os.path.exists(log_file):
                log_file = os.path.join(
                    task_output_dir, 
                    f"{task_name}_importance_prune{prune_percent}_type{importance_type}.log"
                )
            # Try a pattern-based search as a fallback
            if not os.path.exists(log_file):
                pattern = f"{task_name}_importance_prune*_type{importance_type}.log"
                matching_files = glob.glob(os.path.join(task_output_dir, pattern))
                if matching_files:
                    log_file = matching_files[0]
            
            if os.path.exists(log_file):
                # Parse results from the log file
                task_results = parse_log_file(log_file)
                all_results[task_name] = task_results
            else:
                logger.warning(f"Log file not found for task {task_name}")
                pattern_path = os.path.join(task_output_dir, f"{task_name}_importance_prune*_type{importance_type}.log")
                logger.warning(f"Tried looking for: {pattern_path}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running importance pruning for task {task_name}: {e}")
            continue
    
    # Create summary of results
    if all_results:
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY OF RESULTS")
        logger.info("=" * 80)
        
        # Create DataFrame for test results
        test_metrics = {}
        for task, metrics in all_results.items():
            test_metrics[task] = {k: v for k, v in metrics.items() if k.startswith("test_")}
        
        test_df = pd.DataFrame.from_dict(test_metrics, orient='index')
        if not test_df.empty:
            # Clean up column names for display
            test_df.columns = [col.replace("test_", "") for col in test_df.columns]
            
            logger.info("\nTest Results:")
            logger.info("\n" + test_df.to_string())
        
        # Create DataFrame for parameter reduction statistics
        param_metrics = {}
        for task, metrics in all_results.items():
            param_metrics[task] = {
                "vocab_reduction": metrics.get("vocab_reduction", 0),
                "total_param_reduction": metrics.get("total_param_reduction", 0),
                "embedding_param_reduction": metrics.get("embedding_param_reduction", 0),
                "model_only_param_reduction": metrics.get("model_only_param_reduction", 0)
            }
        
        param_df = pd.DataFrame.from_dict(param_metrics, orient='index')
        if not param_df.empty and param_df.sum().sum() > 0:  # Check if we have any non-zero reductions
            logger.info("\nParameter Reduction Statistics:")
            logger.info("\n" + param_df.to_string())
        
        # Save summary to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(args.output_dir, f"summary_results_{timestamp}.csv")
        
        # Combine all metrics
        all_metrics = {}
        for task in all_results:
            all_metrics[task] = all_results[task]
        
        summary_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        summary_df.to_csv(summary_file)
        logger.info(f"\nSaved detailed summary to {summary_file}")
    
    logger.info("\nAll tasks completed!")

if __name__ == "__main__":
    main() 