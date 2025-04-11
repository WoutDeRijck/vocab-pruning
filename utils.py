"""
Utility functions for vocabulary pruning.
"""

import logging
import os
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

# Configure logging
logger = logging.getLogger(__name__)

# GLUE task input keys mapping
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# GLUE task metadata
glue_tasks = {
    "cola": {
        "abbr": "CoLA",
        "name": "Corpus of Linguistic Acceptability",
        "description": "Predict whether a sequence is a grammatical English sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Misc.",
        "size": "8.5k",
        "metrics": "Matthews corr.",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [matthews_corrcoef],
        "n_labels": 2,
    },
    "sst2": {
        "abbr": "SST-2",
        "name": "Stanford Sentiment Treebank",
        "description": "Predict the sentiment of a given sentence",
        "task_type": "Single-Sentence Task",
        "domain": "Movie reviews",
        "size": "67k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "mrpc": {
        "abbr": "MRPC",
        "name": "Microsoft Research Paraphrase Corpus",
        "description": "Predict whether two sentences are semantically equivalent",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "News",
        "size": "3.7k",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score, f1_score],
        "n_labels": 2,
    },
    "stsb": {
        "abbr": "SST-B",
        "name": "Semantic Textual Similarity Benchmark",
        "description": "Predict the similarity score for two sentences on a scale from 1 to 5",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Misc.",
        "size": "7k",
        "metrics": "Pearson/Spearman corr.",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [pearsonr, spearmanr],
        "n_labels": 1,
    },
    "qqp": {
        "abbr": "QQP",
        "name": "Quora question pair",
        "description": "Predict if two questions are a paraphrase of one another",
        "task_type": "Similarity and Paraphrase Tasks",
        "domain": "Social QA questions",
        "size": "364k",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["question1", "question2"],
        "target": "label",
        "metric_funcs": [f1_score, accuracy_score],
        "n_labels": 2,
    },
    "mnli-matched": {
        "abbr": "MNLI",
        "name": "Mulit-Genre Natural Language Inference",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation_matched", "test": "test_matched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 3,
    },
    "mnli-mismatched": {
        "abbr": "MNLI",
        "name": "Mulit-Genre Natural Language Inference",
        "description": "Predict whether the premise entails, contradicts or is neutral to the hypothesis",
        "task_type": "Inference Tasks",
        "domain": "Misc.",
        "size": "393k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation_mismatched", "test": "test_mismatched"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 3,
    },
    "qnli": {
        "abbr": "QNLI",
        "name": "Stanford Question Answering Dataset",
        "description": "Predict whether the context sentence contains the answer to the question",
        "task_type": "Inference Tasks",
        "domain": "Wikipedia",
        "size": "105k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["question", "sentence"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "rte": {
        "abbr": "RTE",
        "name": "Recognize Textual Entailment",
        "description": "Predict whether one sentece entails another",
        "task_type": "Inference Tasks",
        "domain": "News, Wikipedia",
        "size": "2.5k",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "wnli": {
        "abbr": "WNLI",
        "name": "Winograd Schema Challenge",
        "description": "Predict if the sentence with the pronoun substituted is entailed by the original sentence",
        "task_type": "Inference Tasks",
        "domain": "Fiction books",
        "size": "634",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
}

def get_task_metadata(task_name):
    """
    Get metadata for a GLUE task.
    
    Args:
        task_name: Name of the GLUE task
        
    Returns:
        Dictionary with task metadata
    """
    # Handle 'mnli' task name specially
    task_key = "mnli-matched" if task_name == "mnli" else task_name
    return glue_tasks[task_key]

def setup_logging(args):
    """
    Set up logging configuration.
    
    Args:
        args: Command line arguments
    """
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure log file for this run
    log_filename = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}.log"
    if args.pruning_method == "clustering":
        log_filename = f"{args.output_dir}/{args.task}_clustering_prune{args.prune_percent}_{args.clustering_method}.log"
    elif args.pruning_method == "hybrid":
        log_filename = f"{args.output_dir}/{args.task}_hybrid_prune{args.prune_percent}_clusters{args.num_clusters}.log"
    elif args.pruning_method == "importance":
        log_filename = f"{args.output_dir}/{args.task}_importance_prune{args.prune_percent}_clusters{args.num_clusters}_type{args.importance_type}.log"
    
    # Add file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log initial information
    logger.info(f"Starting pruning run with arguments: {args}")
    logger.info(f"Logging to: {log_filename}")

def set_seed(seed):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Optional: for complete reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 