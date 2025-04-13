#!/usr/bin/env python
# coding: utf-8

"""
Clustering-based Vocabulary Pruning Script

This script runs clustering-based vocabulary pruning on all GLUE benchmark tasks.
Each task has customized hyperparameters for optimal performance.
"""

import argparse
import logging
import subprocess
import os
import re
import pandas as pd
from datetime import datetime

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

BATCH_SIZE = 128
CLUSTERING_METHOD = "agglomerative"
PRUNE_PERCENT = 20

# Dictionary of task-specific parameters
TASK_PARAMS = {
    "cola": {
        "epochs": 5,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-6,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "mnli": {
        "epochs": 1,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "mrpc": {
        "epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "qnli": {
        "epochs": 2,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "qqp": {
        "epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "rte": {
        "epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-5,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "sst2": {
        "epochs": 2,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-5,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "stsb": {
        "epochs": 10,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
    "wnli": {
        "epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-5,
        "prune_percent": PRUNE_PERCENT,
        "clustering_method": CLUSTERING_METHOD,
    },
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run clustering-based vocabulary pruning on GLUE tasks"
    )
    
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
        "--output_dir", 
        type=str, 
        default="./clustering_pruning_output",
        help="Output directory for model checkpoints and logs"
    )
    
    parser.add_argument(
        "--prune_percent", 
        type=float, 
        default=None,
        help="Override the default prune percentage for all tasks"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Override the default number of epochs for all tasks"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None,
        help="Override the default learning rate for all tasks"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Override the default batch size for all tasks"
    )
    
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=None,
        help="Override the default weight decay for all tasks"
    )
    
    parser.add_argument(
        "--clustering_method", 
        type=str, 
        default=None,
        choices=["agglomerative", "kmeans"],
        help="Override the default clustering method for all tasks"
    )
    
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
    """Run clustering-based pruning on specified GLUE tasks."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    
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
        if args.clustering_method is not None:
            task_params["clustering_method"] = args.clustering_method
        
        # Create task-specific output directory
        task_output_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Log task parameters
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running clustering-based pruning for task: {task}")
        logger.info(f"Parameters: {task_params}")
        logger.info(f"{'=' * 50}")
        
        # Build command for the main script
        cmd = [
            "python", MAIN_SCRIPT_PATH,
            "--task", task,
            "--model_name", args.model_name,
            "--pruning_method", "clustering",
            "--prune_percent", str(task_params["prune_percent"]),
            "--epochs", str(task_params["epochs"]),
            "--learning_rate", str(task_params["learning_rate"]),
            "--batch_size", str(task_params["batch_size"]),
            "--weight_decay", str(task_params["weight_decay"]),
            "--clustering_method", task_params["clustering_method"],
            "--output_dir", task_output_dir,
            "--seed", str(args.seed),
        ]
        
        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully completed clustering-based pruning for task: {task}")
            
            # Find the log file for this task
            log_file = os.path.join(
                task_output_dir, 
                f"{task}_clustering_prune{task_params['prune_percent']}_{task_params['clustering_method']}.log"
            )
            if os.path.exists(log_file):
                # Parse results from the log file
                task_results = parse_log_file(log_file)
                all_results[task] = task_results
            else:
                logger.warning(f"Log file not found for task {task}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running clustering-based pruning for task {task}: {e}")
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