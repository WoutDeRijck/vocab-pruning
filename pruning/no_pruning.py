"""
No-pruning baseline for vocabulary pruning.

This module implements the baseline approach that maintains the original vocabulary.
"""

import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import get_task_metadata, task_to_keys

# Configure logging
logger = logging.getLogger(__name__)

def setup_no_pruning_model(task_name, model_name):
    """
    Setup a model without any vocabulary pruning (baseline).
    
    Args:
        task_name: Name of the GLUE task
        model_name: Pretrained model name or path
        
    Returns:
        model: Model with original vocabulary
        None: No token map (we keep all tokens)
        None: No OOV lookup (we keep all tokens)
    """
    logger.info(f"Setting up model without vocabulary pruning: {model_name}")
    
    # Get task metadata
    task_meta = get_task_metadata(task_name)
    n_labels = task_meta["n_labels"]
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model for fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels
    )
    
    logger.info(f"Original vocabulary size: {len(tokenizer.get_vocab())}")
    logger.info("No vocabulary pruning applied (baseline)")
    
    # Return the model without any token mapping or OOV lookup
    return model, None, None

def prepare_standard_datasets(task_name, tokenizer, max_length=128):
    """
    Prepare standard datasets for no-pruning baseline.
    Uses the standard tokenizer processing, similar to the notebook.
    
    Args:
        task_name: Name of the GLUE task
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        
    Returns:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Test dataset (if available)
    """
    # Get task metadata
    task_meta = get_task_metadata(task_name)
    train_ds_name = task_meta["dataset_names"]["train"]
    valid_ds_name = task_meta["dataset_names"]["valid"]
    
    # Load dataset
    dataset_name = "mnli" if task_name.startswith("mnli") else task_name
    raw_datasets = load_dataset("glue", dataset_name)
    
    # Get key column names
    sentence1_key, sentence2_key = task_to_keys[dataset_name]
    
    # Define preprocessing function (similar to notebook)
    def preprocess_function(examples):
        if sentence2_key is None:
            inputs = [examples[sentence1_key]]
        else:
            inputs = [examples[sentence1_key], examples[sentence2_key]]
            
        tokenized = tokenizer(*inputs, truncation=True, max_length=max_length)
        return tokenized
    
    # Process datasets
    tokenized_datasets = raw_datasets.map(
        preprocess_function, 
        batched=True,
        desc="Tokenizing datasets"
    )
    
    # Add labels column
    if "label" in tokenized_datasets[train_ds_name].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Get datasets
    train_dataset = tokenized_datasets[train_ds_name]
    eval_dataset = tokenized_datasets[valid_ds_name]
    
    # Get test dataset if available
    test_ds_name = task_meta["dataset_names"].get("test")
    test_dataset = tokenized_datasets[test_ds_name] if test_ds_name in tokenized_datasets else None
    
    return train_dataset, eval_dataset, test_dataset 