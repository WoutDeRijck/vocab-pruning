#!/usr/bin/env python
# coding: utf-8

"""
Frequency-based OOV Vocabulary Pruning Script

This script runs frequency-based OOV vocabulary pruning on all GLUE benchmark tasks.
Each task has customized hyperparameters for optimal performance.

Frequency-OOV pruning combines frequency-based pruning with clustering for OOV tokens.
"""

import argparse
import logging
import subprocess
import os

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

BATCH_SIZE = 64
PRUNE_PERCENT = 20
NUM_CLUSTERS = 50

# Dictionary of task-specific parameters
TASK_PARAMS = {
    "cola": {
        "epochs": 5,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-6,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "mnli": {
        "epochs": 1,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "mrpc": {
        "epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "qnli": {
        "epochs": 2,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "qqp": {
        "epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "rte": {
        "epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-5,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "sst2": {
        "epochs": 2,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-5,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "stsb": {
        "epochs": 10,
        "learning_rate": 8e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 5e-6,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
    "wnli": {
        "epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": BATCH_SIZE,
        "weight_decay": 1e-5,
        "prune_percent": PRUNE_PERCENT,
        "num_clusters": NUM_CLUSTERS,
    },
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run frequency-based OOV vocabulary pruning on GLUE tasks"
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
        default="./frequency_oov_pruning_output",
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
        "--num_clusters", 
        type=int, 
        default=None,
        help="Override the default number of clusters for all tasks"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def main():
    """Run frequency-based OOV pruning on specified GLUE tasks."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    
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
        if args.num_clusters is not None:
            task_params["num_clusters"] = args.num_clusters
        
        # Create task-specific output directory
        task_output_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Log task parameters
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Running frequency-based OOV pruning for task: {task}")
        logger.info(f"Parameters: {task_params}")
        logger.info(f"{'=' * 50}")
        
        # Build command for the main script
        cmd = [
            "python", MAIN_SCRIPT_PATH,
            "--task", task,
            "--model_name", args.model_name,
            "--pruning_method", "frequency_oov",
            "--prune_percent", str(task_params["prune_percent"]),
            "--epochs", str(task_params["epochs"]),
            "--learning_rate", str(task_params["learning_rate"]),
            "--batch_size", str(task_params["batch_size"]),
            "--weight_decay", str(task_params["weight_decay"]),
            "--num_clusters", str(task_params["num_clusters"]),
            "--output_dir", task_output_dir,
            "--seed", str(args.seed),
        ]
        
        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully completed frequency-based OOV pruning for task: {task}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running frequency-based OOV pruning for task {task}: {e}")
            continue
    
    logger.info("\nAll tasks completed!")

if __name__ == "__main__":
    main() 