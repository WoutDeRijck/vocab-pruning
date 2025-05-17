#!/usr/bin/env python
# coding: utf-8

"""
Attention-based Vocabulary Pruning Example Script

This script demonstrates how to use the attention-based vocabulary pruning technique
on GLUE benchmark tasks. This approach uses attention patterns from a fine-tuned model
to determine token importance in context.

Example usage:
    python run_attention_pruning.py --task mrpc --prune_percent 20 --no_pruning_dir path/to/no_pruning_output
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

# Default settings for each task (customize as needed)
TASK_PARAMS = {
    "cola": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 8,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 1e-6
    },
    "mnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 1,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 5e-6
    },
    "mrpc": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 15,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 5e-6
    },
    "qnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 2,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 5e-6
    },
    "qqp": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 5e-6
    },
    "rte": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 1e-5
    },
    "sst2": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 2,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 1e-5
    },
    "stsb": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 8e-5, 
        "epochs": 10,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 5e-6
    },
    "wnli": {
        "batch_size": BATCH_SIZE, 
        "learning_rate": 5e-5, 
        "epochs": 3,
        "prune_percent": PRUNE_PERCENT,
        "weight_decay": 1e-5
    }
}

def find_finetuned_model(no_pruning_dir, task_name):
    """
    Find the fine-tuned model checkpoint for a specific task in the no-pruning directory.
    
    Args:
        no_pruning_dir: Directory containing no-pruning results
        task_name: Name of the GLUE task
        
    Returns:
        Path to the fine-tuned model checkpoint, or None if not found
    """
    # Check if task directory exists
    task_dir = os.path.join(no_pruning_dir, task_name)
    if not os.path.exists(task_dir):
        logger.warning(f"No task directory found at {task_dir}")
        return None
    
    # Look for model checkpoint directory - model checkpoints are directories with model files
    checkpoint_dirs = []
    for item in os.listdir(task_dir):
        item_path = os.path.join(task_dir, item)
        # Check if it's a directory that looks like a model checkpoint
        if os.path.isdir(item_path) and any([
            os.path.exists(os.path.join(item_path, file))
            for file in ["config.json", "pytorch_model.bin", "model.safetensors"]
        ]):
            checkpoint_dirs.append(item_path)
    
    if not checkpoint_dirs:
        logger.warning(f"No model checkpoint directories found in {task_dir}")
        return None
    
    # Sort by creation time (newest first)
    checkpoint_dirs.sort(key=os.path.getctime, reverse=True)
    model_dir = checkpoint_dirs[0]
    
    logger.info(f"Found fine-tuned model checkpoint: {model_dir}")
    return model_dir

def parse_args():
    """Parse command line arguments for Attention-based pruning."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models with Attention-based vocabulary pruning"
    )
    
    # Task and model arguments
    parser.add_argument(
        "--tasks", 
        type=str, 
        nargs="+",
        default=list(TASK_PARAMS.keys()),
        help="GLUE tasks to run (default: all)"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name or path"
    )
    
    parser.add_argument(
        "--finetuned_model_path", 
        type=str, 
        default=None,
        help="Path to a fine-tuned model to use for attention calculation (optional)"
    )
    
    parser.add_argument(
        "--no_pruning_dir", 
        type=str, 
        default=None,
        help="Directory containing fine-tuned models from no-pruning runs"
    )
    
    # Pruning arguments
    parser.add_argument(
        "--prune_percent", 
        type=float, 
        default=None,
        help="Percentage of vocabulary to prune (overrides task defaults)"
    )
    
    # Output directory
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./attention_pruning_output",
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
    
    parser.add_argument(
        "--param_based", 
        action="store_true",
        help="If set, prune based on parameter percentage rather than token percentage"
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
    """Run attention-based pruning on specified GLUE tasks."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a subdirectory based on pruning type
    pruning_type = "param_based" if args.param_based else "token_based"
    pruning_dir = os.path.join(args.output_dir, pruning_type)
    os.makedirs(pruning_dir, exist_ok=True)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    logger.info(f"Pruning type: {'Parameter-based' if args.param_based else 'Token-based'}")
    
    # Dictionary to store results for each task
    all_results = {}
    
    # Run each task
    for task in args.tasks:
        if task not in TASK_PARAMS:
            logger.warning(f"Task {task} not found in task parameters, skipping.")
            continue
        
        # Get task parameters
        task_params = TASK_PARAMS[task].copy()
        
        # Override task parameters if specified
        if args.prune_percent is not None:
            task_params["prune_percent"] = args.prune_percent
        if args.epochs is not None:
            task_params["epochs"] = args.epochs
        if args.learning_rate is not None:
            task_params["learning_rate"] = args.learning_rate
        if args.batch_size is not None:
            task_params["batch_size"] = args.batch_size
        if args.weight_decay is not None:
            task_params["weight_decay"] = args.weight_decay
        
        # Create task-specific output directory
        task_output_dir = os.path.join(pruning_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Determine fine-tuned model to use
        finetuned_model = args.finetuned_model_path
        
        # If no explicit fine-tuned model path is provided, try to find one from no_pruning_dir
        if not finetuned_model and args.no_pruning_dir:
            finetuned_model = find_finetuned_model(args.no_pruning_dir, task)
            if not finetuned_model:
                logger.warning(f"No fine-tuned model found for task {task} in {args.no_pruning_dir}")
                logger.info(f"Falling back to base model: {args.model_name}")
        
        # Determine which model to use for embeddings and attention calculation
        model_name = args.model_name
        attention_model = finetuned_model if finetuned_model else args.model_name
        
        # Log task parameters
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running attention-based pruning for task: {task}")
        logger.info(f"Parameters: {task_params}")
        logger.info(f"Using model: {model_name}")
        logger.info(f"Using attention model: {attention_model}")
        logger.info(f"{'=' * 50}")
        
        # Build command for the main script
        cmd = [
            "python", MAIN_SCRIPT_PATH,
            "--task", task,
            "--model_name", model_name,
            "--pruning_method", "attention",
            "--attention_model", attention_model if attention_model != model_name else "",
            "--prune_percent", str(task_params["prune_percent"]),
            "--epochs", str(task_params["epochs"]),
            "--learning_rate", str(task_params["learning_rate"]),
            "--batch_size", str(task_params["batch_size"]),
            "--weight_decay", str(task_params["weight_decay"]),
            "--output_dir", task_output_dir,
            "--seed", str(args.seed),
        ]
        
        # Remove empty arguments
        cmd = [arg for arg in cmd if arg]
        
        # Add param_based flag if needed
        if args.param_based:
            cmd.append("--param_based")
        
        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully completed attention-based pruning for task: {task}")
            
            # Find the log file for this task
            suffix = "param" if args.param_based else "token"
            log_file = os.path.join(
                task_output_dir, 
                f"{task}_attention_{suffix}_prune{int(task_params['prune_percent'])}.log"
            )
            # Try alternative filename if not found
            if not os.path.exists(log_file):
                log_file = os.path.join(
                    task_output_dir, 
                    f"{task}_attention_prune{task_params['prune_percent']}.log"
                )
            # Try a pattern-based search as a fallback
            if not os.path.exists(log_file):
                pattern = f"{task}_attention*prune*.log"
                matching_files = glob.glob(os.path.join(task_output_dir, pattern))
                if matching_files:
                    log_file = matching_files[0]
            
            if os.path.exists(log_file):
                # Parse results from the log file
                task_results = parse_log_file(log_file)
                all_results[task] = task_results
            else:
                logger.warning(f"Log file not found for task {task}")
                pattern_path = os.path.join(task_output_dir, f"{task}_attention*prune*.log")
                logger.warning(f"Tried looking for: {pattern_path}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running attention-based pruning for task {task}: {e}")
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