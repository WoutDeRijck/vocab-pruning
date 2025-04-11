"""
Split utilities for GLUE tasks.

This module provides functions to create consistent train/validation/test splits
from the original GLUE train and validation sets.
"""

import logging
from typing import Tuple, Generator

import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Task to keys mapping from the original code
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

def create_consistent_splits(
    task_name: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create consistent train/validation/test splits from GLUE datasets.
    
    Args:
        task_name: Name of the GLUE task
        train_ratio: Proportion of data to use for training
        validation_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    # Validate ratios sum to 1
    total_ratio = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Handle 'mnli' special case
    dataset_name = "mnli" if task_name.startswith("mnli") else task_name
    logger.info(f"Loading {dataset_name} dataset")
    
    # Load original train and validation datasets
    raw_datasets = load_dataset("glue", dataset_name)
    
    # Get split names for train and validation
    train_split_name = "train"
    
    # For MNLI, the validation set is either validation_matched or validation_mismatched
    if task_name == "mnli-matched":
        valid_split_name = "validation_matched"
    elif task_name == "mnli-mismatched":
        valid_split_name = "validation_mismatched"
    else:
        valid_split_name = "validation"
    
    # Get original train and validation sets
    train_dataset = raw_datasets[train_split_name]
    valid_dataset = raw_datasets[valid_split_name]
    
    logger.info(f"Original train set size: {len(train_dataset)}")
    logger.info(f"Original validation set size: {len(valid_dataset)}")
    
    # Combine train and validation sets
    combined_dataset = concatenate_datasets([train_dataset, valid_dataset])
    logger.info(f"Combined dataset size: {len(combined_dataset)}")
    
    # Check if the dataset has a 'label' column
    if 'label' in combined_dataset.column_names:
        # Get labels for stratification
        labels = combined_dataset['label']
        
        # For regression tasks like STS-B, we can't do stratified splitting
        is_regression = task_name == "stsb"
        stratify = None if is_regression else labels
        
        # Create a temporary split first to get training set
        temp_train, temp_testval, train_idx, testval_idx = train_test_split(
            np.arange(len(combined_dataset)),
            np.arange(len(combined_dataset)),
            train_size=train_ratio,
            random_state=random_seed,
            stratify=stratify
        )
        
        # Recalculate stratify for test/val split if we're doing stratification
        if stratify is not None:
            stratify = [labels[i] for i in testval_idx]
        
        # Now split the temp_testval into validation and test sets
        val_ratio_adjusted = validation_ratio / (validation_ratio + test_ratio)
        _, _, val_idx, test_idx = train_test_split(
            np.arange(len(testval_idx)),
            np.arange(len(testval_idx)),
            train_size=val_ratio_adjusted,
            random_state=random_seed,
            stratify=stratify
        )
        
        # Convert temporary indices to original dataset indices
        final_val_idx = [testval_idx[i] for i in val_idx]
        final_test_idx = [testval_idx[i] for i in test_idx]
        
        # Create final datasets using selected indices
        final_train_dataset = combined_dataset.select(train_idx)
        final_valid_dataset = combined_dataset.select(final_val_idx)
        final_test_dataset = combined_dataset.select(final_test_idx)
    else:
        # If no label column, just do a random split
        logger.warning(f"No 'label' column found in dataset, using non-stratified split")
        
        # First split train vs. (valid+test)
        train_idx, testval_idx = train_test_split(
            np.arange(len(combined_dataset)),
            train_size=train_ratio,
            random_state=random_seed
        )
        
        # Calculate adjusted validation ratio from the test+validation pool
        val_ratio_adjusted = validation_ratio / (validation_ratio + test_ratio)
        
        # Split (valid+test) into valid and test
        val_idx, test_idx = train_test_split(
            testval_idx,
            train_size=val_ratio_adjusted,
            random_state=random_seed
        )
        
        # Create final datasets
        final_train_dataset = combined_dataset.select(train_idx)
        final_valid_dataset = combined_dataset.select(val_idx)
        final_test_dataset = combined_dataset.select(test_idx)
    
    # Log resulting split sizes
    logger.info(f"New train set size: {len(final_train_dataset)} ({len(final_train_dataset)/len(combined_dataset)*100:.1f}%)")
    logger.info(f"New validation set size: {len(final_valid_dataset)} ({len(final_valid_dataset)/len(combined_dataset)*100:.1f}%)")
    logger.info(f"New test set size: {len(final_test_dataset)} ({len(final_test_dataset)/len(combined_dataset)*100:.1f}%)")
    
    # For classification tasks, verify label distribution
    if 'label' in combined_dataset.column_names and task_name != "stsb":
        # Count original labels
        orig_label_counts = {}
        for label in combined_dataset['label']:
            orig_label_counts[label] = orig_label_counts.get(label, 0) + 1
        
        # Count labels in each split
        train_label_counts = {}
        for label in final_train_dataset['label']:
            train_label_counts[label] = train_label_counts.get(label, 0) + 1
            
        valid_label_counts = {}
        for label in final_valid_dataset['label']:
            valid_label_counts[label] = valid_label_counts.get(label, 0) + 1
            
        test_label_counts = {}
        for label in final_test_dataset['label']:
            test_label_counts[label] = test_label_counts.get(label, 0) + 1
        
        # Log label distributions
        logger.info("Label distribution:")
        for label in sorted(orig_label_counts.keys()):
            orig_pct = orig_label_counts[label] / len(combined_dataset) * 100
            train_pct = train_label_counts.get(label, 0) / len(final_train_dataset) * 100
            valid_pct = valid_label_counts.get(label, 0) / len(final_valid_dataset) * 100
            test_pct = test_label_counts.get(label, 0) / len(final_test_dataset) * 100
            
            logger.info(f"  Label {label}: original={orig_pct:.1f}%, train={train_pct:.1f}%, valid={valid_pct:.1f}%, test={test_pct:.1f}%")
    
    return final_train_dataset, final_valid_dataset, final_test_dataset

def prepare_custom_split_datasets(
    task_name: str,
    tokenizer,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_length: int = 128,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create custom splits and tokenize them ready for model training.
    
    Args:
        task_name: Name of the GLUE task
        tokenizer: Tokenizer to use
        train_ratio: Proportion of data to use for training
        validation_ratio: Proportion of data to use for validation
        test_ratio: Proportion of data to use for testing
        max_length: Maximum sequence length
        random_seed: Random seed
        
    Returns:
        Tuple of prepared (train_dataset, validation_dataset, test_dataset)
    """
    # First create the splits
    train_dataset, valid_dataset, test_dataset = create_consistent_splits(
        task_name=task_name,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Get input keys for this task
    sentence1_key, sentence2_key = task_to_keys[task_name.replace("-matched", "").replace("-mismatched", "")]
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Handle single sentence or sentence pairs
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    # Apply preprocessing to all splits
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing train dataset",
    )
    
    valid_dataset = valid_dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing validation dataset",
    )
    
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Tokenizing test dataset",
    )
    
    # Rename 'label' to 'labels' to match what the model expects
    if 'label' in train_dataset.column_names:
        train_dataset = train_dataset.rename_column('label', 'labels')
        valid_dataset = valid_dataset.rename_column('label', 'labels')
        test_dataset = test_dataset.rename_column('label', 'labels')
    else:
        logger.warning("No 'label' column found in datasets")
    
    return train_dataset, valid_dataset, test_dataset

def create_cross_validation_splits(
    task_name: str,
    n_folds: int = 5,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Generator[Tuple[Dataset, Dataset, Dataset], None, None], Dataset]:
    """
    Create cross-validation splits from GLUE datasets, plus a separate test set.
    
    Args:
        task_name: Name of the GLUE task
        n_folds: Number of folds for cross-validation
        test_ratio: Proportion of data to use for the held-out test set
        random_seed: Random seed for reproducibility
        
    Returns:
        cv_splits: Generator yielding (train, val) datasets for each fold
        test_dataset: The common test dataset (same for all folds)
    """
    # First separate out a test set
    logger.info(f"Creating {n_folds}-fold cross-validation splits with {test_ratio:.0%} test set for {task_name}")
    
    # Handle 'mnli' special case
    dataset_name = "mnli" if task_name.startswith("mnli") else task_name
    
    # Load original train and validation datasets
    raw_datasets = load_dataset("glue", dataset_name)
    
    # Get split names for train and validation
    train_split_name = "train"
    
    # For MNLI, the validation set is either validation_matched or validation_mismatched
    if task_name == "mnli-matched":
        valid_split_name = "validation_matched"
    elif task_name == "mnli-mismatched":
        valid_split_name = "validation_mismatched"
    else:
        valid_split_name = "validation"
    
    # Get original train and validation sets
    train_dataset = raw_datasets[train_split_name]
    valid_dataset = raw_datasets[valid_split_name]
    
    logger.info(f"Original train set size: {len(train_dataset)}")
    logger.info(f"Original validation set size: {len(valid_dataset)}")
    
    # Combine train and validation sets
    combined_dataset = concatenate_datasets([train_dataset, valid_dataset])
    logger.info(f"Combined dataset size: {len(combined_dataset)}")
    
    # First separate out the test set
    is_regression = task_name == "stsb"
    stratify = None if is_regression else combined_dataset['label'] if 'label' in combined_dataset.column_names else None
    
    train_val_indices, test_indices = train_test_split(
        np.arange(len(combined_dataset)),
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify
    )
    
    # Create the test dataset
    test_dataset = combined_dataset.select(test_indices)
    logger.info(f"Test set size: {len(test_dataset)} ({len(test_dataset)/len(combined_dataset)*100:.1f}%)")
    
    # Get the training+validation set
    train_val_dataset = combined_dataset.select(train_val_indices)
    
    # Set up cross-validation
    if is_regression or 'label' not in train_val_dataset.column_names:
        # For regression or if no labels, use KFold
        logger.info("Using KFold cross-validation (non-stratified)")
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_indices = list(cv.split(np.arange(len(train_val_dataset))))
    else:
        # For classification, use StratifiedKFold
        logger.info("Using StratifiedKFold cross-validation")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_indices = list(cv.split(
            np.arange(len(train_val_dataset)), 
            train_val_dataset['label']
        ))
    
    def generate_fold_splits():
        for fold_idx, (train_idx, val_idx) in enumerate(split_indices):
            logger.info(f"Fold {fold_idx+1}/{n_folds} - train: {len(train_idx)} examples, val: {len(val_idx)} examples")
            fold_train = train_val_dataset.select(train_idx)
            fold_val = train_val_dataset.select(val_idx)
            yield fold_train, fold_val, test_dataset
    
    return generate_fold_splits(), test_dataset

def prepare_cross_validation_datasets(
    task_name: str,
    tokenizer,
    n_folds: int = 5,
    test_ratio: float = 0.1,
    max_length: int = 128,
    random_seed: int = 42
) -> Generator[Tuple[Dataset, Dataset, Dataset], None, None]:
    """
    Create and tokenize cross-validation splits ready for model training.
    
    Args:
        task_name: Name of the GLUE task
        tokenizer: Tokenizer to use
        n_folds: Number of folds for cross-validation
        test_ratio: Proportion of data to use for the held-out test set
        max_length: Maximum sequence length
        random_seed: Random seed
        
    Returns:
        Generator yielding tokenized (train, val, test) datasets for each fold
    """
    # Get the fold splits
    fold_splits_generator, _ = create_cross_validation_splits(
        task_name=task_name,
        n_folds=n_folds,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # Get input keys for this task
    sentence1_key, sentence2_key = task_to_keys[task_name.replace("-matched", "").replace("-mismatched", "")]
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Handle single sentence or sentence pairs
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    # Process each fold
    for fold_idx, (train_dataset, val_dataset, test_dataset) in enumerate(fold_splits_generator):
        logger.info(f"Tokenizing fold {fold_idx+1}/{n_folds}")
        
        # Apply preprocessing to all splits
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc=f"Tokenizing fold {fold_idx+1} train",
        )
        
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            desc=f"Tokenizing fold {fold_idx+1} validation",
        )
        
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            desc=f"Tokenizing fold {fold_idx+1} test",
        )
        
        # Rename 'label' to 'labels' to match what the model expects
        if 'label' in train_dataset.column_names:
            train_dataset = train_dataset.rename_column('label', 'labels')
            val_dataset = val_dataset.rename_column('label', 'labels')
            test_dataset = test_dataset.rename_column('label', 'labels')
        else:
            logger.warning(f"Fold {fold_idx+1}: No 'label' column found in datasets")
        
        yield train_dataset, val_dataset, test_dataset 