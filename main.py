#!/usr/bin/env python
# coding: utf-8

"""
Modular Vocabulary Pruning Script

This script uses the modular pruning package to run vocabulary pruning
experiments on GLUE benchmark tasks with various techniques:
1. Clustering-based pruning: Group similar tokens and keep representatives
2. Frequency-based pruning: Remove least frequently used tokens
3. Hybrid pruning: Combine frequency-based pruning with clustering for OOV tokens
4. Word importance pruning: Use TF-IDF to determine token importance
5. Random selection pruning: Randomly select tokens to keep or remove (baseline)
6. Word importance pruning without OOV: Use TF-IDF with OOV tokens mapped to UNK
7. Attention-based pruning: Use attention patterns from a fine-tuned model to determine token importance
8. Train-only pruning: Only keep tokens that appear in the training set (baseline)
9. No-pruning baseline: Keep the full original vocabulary without any changes (baseline)
"""

import os
import argparse
import logging
import copy
import traceback

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from sklearn.metrics import precision_score, recall_score

# Import pruning modules
from pruning.clustering import setup_clustering_based_model
from pruning.frequency import setup_frequency_based_model
from pruning.frequency_oov import setup_frequency_oov_model
from pruning.importance_oov import setup_importance_oov_model
from pruning.importance import setup_importance_based_model
from pruning.random import setup_random_based_model
from pruning.attention import setup_attention_based_model
from pruning.train_only import setup_train_only_model
from pruning.no_pruning import setup_no_pruning_model
from pruning.base import task_to_keys
from utils import glue_tasks

# Import functions from data.py and training.py
from training import setup_training, generate_test_predictions
from metrics import count_model_parameters

# Disable PyTorch dynamo compiler to avoid CUDA illegal memory access errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate models with modular vocabulary pruning"
    )
    
    # Common arguments
    parser.add_argument(
        "--task", 
        type=str, 
        default="sst2", 
        choices=list(task_to_keys.keys()),
        help="GLUE task name"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="answerdotai/ModernBERT-base",
        help="Pretrained model name or path"
    )
    
    parser.add_argument(
        "--pruning_method", 
        type=str, 
        default="clustering",
        choices=["clustering", "frequency", "frequency_oov", "importance_oov", "importance", "random", "attention", "train_only", "no_pruning"],
        help="Method to use for vocabulary pruning"
    )
    
    parser.add_argument(
        "--prune_percent", 
        type=float, 
        default=20,
        help="Percentage of vocabulary to prune"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=8e-5,
        help="Learning rate"
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
        default=32,
        help="Training batch size"
    )
    
    # Clustering method specific arguments
    parser.add_argument(
        "--clustering_method", 
        type=str, 
        default="agglomerative",
        choices=["agglomerative", "kmeans"],
        help="Clustering algorithm (for clustering method)"
    )
    
    # Hybrid method specific arguments
    parser.add_argument(
        "--num_clusters", 
        type=int, 
        default=50,
        help="Number of clusters for OOV token mapping (for frequency_oov and importance_oov methods)"
    )
    
    # Word importance specific arguments
    parser.add_argument(
        "--importance_type", 
        type=int, 
        default=3,
        choices=[0, 1, 2, 3],
        help="Word importance calculation type: 0=off, 1=no norm, 2=L1 norm, 3=L2 norm (default)"
    )
    
    # Attention-based pruning specific arguments
    parser.add_argument(
        "--attention_model", 
        type=str, 
        default=None,
        help="Fine-tuned model to use for attention-based pruning (defaults to model_name)"
    )
    
    # Cross-validation related arguments
    parser.add_argument(
        "--cross_validation", 
        action="store_true",
        help="Use cross-validation instead of fixed train/val/test split"
    )
    
    parser.add_argument(
        "--n_folds", 
        type=int, 
        default=5,
        help="Number of folds for cross-validation"
    )
    
    # Custom split related arguments - always used now
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
    
    # Common optional arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./modular_model_output",
        help="Output directory for model checkpoints and logs"
    )
    
    parser.add_argument(
        "--train_only", 
        action="store_true",
        help="Use only the training set for vocabulary extraction"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()

def create_results_dataframe(metrics_callback):
    """Helper function to create a DataFrame from metrics callback."""
    try:
        # Check if we have training history
        if metrics_callback.training_history["train"] and metrics_callback.training_history["eval"]:
            # Handle potential arrays with different lengths
            train_data = metrics_callback.training_history["train"]
            
            # Find min length of all arrays
            min_len = min(len(arr) for arr in train_data.values())
            
            # Truncate all arrays to the same length
            train_data_truncated = {k: v[:min_len] for k, v in train_data.items()}
            
            # Create DataFrames
            train_history_df = pd.DataFrame(train_data_truncated)
            train_history_df = train_history_df.add_prefix("train_")
            
            # Do the same for eval metrics
            eval_data = metrics_callback.training_history["eval"]
            min_len_eval = min(len(arr) for arr in eval_data.values() if len(arr) > 0)
            eval_data_truncated = {k: v[:min_len_eval] for k, v in eval_data.items() if len(v) > 0}
            
            if eval_data_truncated:
                eval_history_df = pd.DataFrame(eval_data_truncated)
                # Combine only if there's data to combine
                if not eval_history_df.empty and not train_history_df.empty:
                    results_df = pd.concat([train_history_df, eval_history_df], axis=1)
                else:
                    results_df = train_history_df if not train_history_df.empty else eval_history_df
            else:
                results_df = train_history_df
            
            return results_df
        else:
            logger.warning("No training history available, skipping results DataFrame creation")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error creating results DataFrame: {e}")
        logger.warning("Continuing without saving training history")
        return pd.DataFrame()

def evaluate_test_set(model, test_dataset, task_name, data_collator, batch_size, pruning_method="clustering", tokenizer=None, trainer=None):
    """
    Evaluate model on test set and return metrics.
    
    Args:
        model: The trained model
        test_dataset: Test dataset with labels
        task_name: Name of the GLUE task
        data_collator: Data collator to use
        batch_size: Batch size for evaluation
        pruning_method: Pruning method used
        tokenizer: Tokenizer (for no_pruning case)
        trainer: The trainer object (if available) for consistent prediction
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Generating predictions on test set")
    
    # If trainer is provided, use it for predictions to ensure consistent processing
    if trainer is not None:
        logger.info("Using Trainer.predict() for consistent processing")
        prediction_output = trainer.predict(test_dataset)
        predictions = prediction_output.predictions
        
        # For classification tasks, get the class with highest probability
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        else:
            # For regression tasks
            predictions = predictions.squeeze()
    else:
        # Fall back to manual processing if no trainer is provided
        with torch.no_grad():
            device = next(model.parameters()).device
            predictions = []
            
            # Special handling for no_pruning with DataCollatorWithPadding
            if pruning_method == "no_pruning" and tokenizer is not None:
                logger.info("Using custom dataloader for no_pruning with standard tokenizer")
                try:
                    # Process one example at a time - safest approach
                    logger.info("Processing examples individually")
                    for i in tqdm(range(len(test_dataset)), desc="Predicting"):
                        example = test_dataset[i]
                        
                        # Extract input_ids and attention_mask, ensuring they're tensors
                        if isinstance(example['input_ids'], torch.Tensor):
                            input_ids = example['input_ids'].unsqueeze(0).to(device)  # Add batch dimension
                        else:
                            # Convert to tensor if it's a list or other type
                            input_ids = torch.tensor([example['input_ids']], dtype=torch.long).to(device)
                        
                        if isinstance(example['attention_mask'], torch.Tensor):
                            attention_mask = example['attention_mask'].unsqueeze(0).to(device)
                        else:
                            attention_mask = torch.tensor([example['attention_mask']], dtype=torch.long).to(device)
                        
                        # Forward pass
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        predictions.append(logits.cpu().numpy())
                except Exception as e:
                    logger.error(f"Error processing examples: {e}")
                    logger.error(traceback.format_exc())
                    return {}
            else:
                # Process in batches using the provided collator
                dataloader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size,
                    collate_fn=data_collator
                )
                
                for batch in tqdm(dataloader, desc="Predicting"):
                    # Filter out non-tensor values and labels
                    batch_inputs = {k: v.to(device) for k, v in batch.items() 
                               if k != 'labels' and k != 'prediction_only' and hasattr(v, 'to')}
                    
                    with torch.no_grad():
                        outputs = model(**batch_inputs)
                    logits = outputs.logits
                    predictions.append(logits.cpu().numpy())
            
            # Concatenate predictions
            all_predictions = np.vstack(predictions)
            
            # For classification tasks, get the class with highest probability
            if len(all_predictions.shape) > 1 and all_predictions.shape[1] > 1:
                predictions = np.argmax(all_predictions, axis=1)
            else:
                # For regression tasks
                predictions = all_predictions.squeeze()
    
    # Save predictions to file
    output_test_file = f"{model.config._name_or_path.replace('/', '_')}_{task_name}_predictions.txt"
    np.savetxt(output_test_file, predictions, fmt='%d' if len(predictions.shape) == 1 else '%.6f')
    logger.info(f"Saved test predictions to {output_test_file}")
    
    # Get the true labels from the test dataset
    true_labels = test_dataset['labels']
    
    # Calculate test metrics
    logger.info("\n=== Test Set Evaluation Results ===")
    
    task_key = "mnli-matched" if task_name == "mnli" else task_name
    task_meta = glue_tasks[task_key]
    
    metrics_dict = {}
    
    for metric_func in task_meta["metric_funcs"]:
        metric_name = metric_func.__name__
        
        # Special handling for correlation metrics which return a tuple
        if metric_name in ['pearsonr', 'spearmanr']:
            result, p_value = metric_func(true_labels, predictions)
            logger.info(f"{metric_name}: {result:.4f}")
            metrics_dict[metric_name] = result
        # Special handling for F1 which needs additional arguments for binary classification
        elif metric_name == 'f1_score' and task_name in ['mrpc', 'qqp']:
            result = metric_func(true_labels, predictions, average='binary')
            logger.info(f"{metric_name}: {result:.4f}")
            metrics_dict[metric_name] = result
        else:
            result = metric_func(true_labels, predictions)
            logger.info(f"{metric_name}: {result:.4f}")
            metrics_dict[metric_name] = result
    
    # For MRPC or QQP, also calculate precision and recall
    if task_name in ['mrpc', 'qqp']:
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        logger.info(f"precision: {precision:.4f}")
        logger.info(f"recall: {recall:.4f}")
        metrics_dict['precision'] = precision
        metrics_dict['recall'] = recall
    
    return metrics_dict

def run_cross_validation_pipeline(args, tokenizer):
    """
    Run the training pipeline with cross-validation.
    
    Args:
        args: Command-line arguments
        tokenizer: Tokenizer for the model
        
    Returns:
        results_df: DataFrame with aggregated results across folds
        best_model: Best performing model across all folds
    """
    try:
        # Import cross-validation utilities
        from split_utils import prepare_cross_validation_datasets
        
        # Create cross-validation datasets
        logger.info(f"Creating {args.n_folds}-fold cross-validation datasets")
        cv_datasets = prepare_cross_validation_datasets(
            task_name=args.task,
            tokenizer=tokenizer,
            n_folds=args.n_folds,
            test_ratio=args.test_ratio,
            max_length=128,
            random_seed=args.seed
        )
        
        # Track results across folds
        fold_results = []
        fold_test_metrics = []
        fold_models = []
        
        # For each fold
        for fold_idx, (train_dataset, eval_dataset, test_dataset) in enumerate(cv_datasets):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Processing fold {fold_idx+1}/{args.n_folds}")
            logger.info(f"{'=' * 50}")
            
            # Set up model with appropriate pruning method for this fold
            if args.pruning_method == "clustering":
                model, token_map, oov_lookup = setup_clustering_based_model(
                    args.task, 
                    args.model_name,
                    prune_percent=args.prune_percent,
                    clustering_method=args.clustering_method
                )
            elif args.pruning_method == "frequency":
                model, token_map, oov_lookup = setup_frequency_based_model(
                    args.task, 
                    args.model_name,
                    prune_percent=args.prune_percent
                )
            elif args.pruning_method == "frequency_oov":
                model, token_map, oov_lookup = setup_frequency_oov_model(
                    args.task, 
                    args.model_name,
                    prune_percent=args.prune_percent,
                    num_clusters=args.num_clusters
                )
            elif args.pruning_method == "importance_oov":
                model, token_map, oov_lookup = setup_importance_oov_model(
                    args.task, 
                    args.model_name,
                    prune_percent=args.prune_percent,
                    num_clusters=args.num_clusters,
                    importance_type=args.importance_type
                )
            elif args.pruning_method == "importance":
                model, token_map, oov_lookup = setup_importance_based_model(
                    args.task, 
                    args.model_name,
                    prune_percent=args.prune_percent,
                    importance_type=args.importance_type
                )
            elif args.pruning_method == "random":
                model, token_map, oov_lookup = setup_random_based_model(
                    args.task, 
                    args.model_name,
                    prune_percent=args.prune_percent,
                    random_seed=args.seed
                )
            elif args.pruning_method == "attention":
                model, token_map, oov_lookup = setup_attention_based_model(
                    args.task, 
                    args.model_name,
                    attention_model=args.attention_model,
                    prune_percent=args.prune_percent
                )
            elif args.pruning_method == "train_only":
                model, token_map, oov_lookup = setup_train_only_model(
                    args.task, 
                    args.model_name
                )
            elif args.pruning_method == "no_pruning":
                model, token_map, oov_lookup = setup_no_pruning_model(
                    args.task, 
                    args.model_name
                )
            
            # Apply token remapping to create datasets with reduced vocabulary
            logger.info("Applying vocabulary mapping to datasets")
            
            # Function to remap tokens in a dataset
            def remap_tokens(dataset):
                # If no token_map (no pruning case), return the dataset unchanged
                if token_map is None:
                    logger.info("No token mapping applied (using full vocabulary)")
                    return dataset
                
                # Create a mapping dictionary for quick lookup
                id_map_dict = token_map.copy()
                if oov_lookup:
                    id_map_dict.update(oov_lookup)
                
                def map_example(example):
                    # Map input_ids using our token mapping
                    new_input_ids = []
                    for token_id in example['input_ids']:
                        if token_id in id_map_dict:
                            new_input_ids.append(id_map_dict[token_id])
                        else:
                            # Use UNK token (0) for OOV tokens
                            new_input_ids.append(0)
                    
                    example['input_ids'] = new_input_ids
                    return example
                
                # Apply mapping to each example
                remapped_dataset = dataset.map(map_example, desc="Remapping token IDs")
                
                # Rename 'label' to 'labels' if it exists (important for data collator compatibility)
                if 'label' in remapped_dataset.column_names and 'labels' not in remapped_dataset.column_names:
                    logger.info("Renaming 'label' column to 'labels' for compatibility with data collator")
                    remapped_dataset = remapped_dataset.rename_column('label', 'labels')
                
                return remapped_dataset
            
            # Apply remapping to all datasets
            remapped_train = remap_tokens(train_dataset)
            remapped_eval = remap_tokens(eval_dataset)
            remapped_test = remap_tokens(test_dataset)
            
            # Set up fold-specific output directory
            fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx+1}")
            os.makedirs(fold_output_dir, exist_ok=True)
            
            # Create a copy of args with the fold output directory
            fold_args = copy.deepcopy(args)
            fold_args.output_dir = fold_output_dir
            
            # Setup trainer for this fold
            if args.pruning_method == "no_pruning":
                trainer, metrics_callback = setup_training(
                    model, 
                    train_dataset,  # Use original dataset
                    eval_dataset,   # Use original dataset
                    args.task, 
                    fold_args,
                    tokenizer=tokenizer
                )
            else:
                trainer, metrics_callback = setup_training(
                    model, 
                    remapped_train, 
                    remapped_eval, 
                    args.task, 
                    fold_args
                )
            
            # Train and evaluate the model for this fold
            logger.info(f"Starting training for fold {fold_idx+1}")
            
            try:
                trainer.train()
                
                # Evaluate the model on validation set
                logger.info(f"Evaluating fold {fold_idx+1} on validation set")
                eval_results = trainer.evaluate()
                
                # Store the model for this fold
                fold_models.append(trainer.model)
                
                # Get training history for this fold
                fold_df = create_results_dataframe(metrics_callback)
                if not fold_df.empty:
                    # Add fold number as a column
                    fold_df['fold'] = fold_idx+1
                    fold_results.append(fold_df)
                
                # Generate predictions on test set
                logger.info(f"Generating test predictions for fold {fold_idx+1}")
                if args.pruning_method == "no_pruning":
                    test_metrics = evaluate_test_set(
                        trainer.model, 
                        remapped_test, 
                        args.task, 
                        trainer.data_collator, 
                        args.batch_size,
                        pruning_method=args.pruning_method,
                        tokenizer=tokenizer,
                        trainer=trainer  # Pass the trainer for consistent processing
                    )
                else:
                    test_metrics = evaluate_test_set(
                        trainer.model, 
                        remapped_test, 
                        args.task, 
                        trainer.data_collator, 
                        args.batch_size,
                        pruning_method=args.pruning_method,
                        trainer=trainer  # Pass the trainer for consistent processing
                    )
                
                # Add fold info to metrics
                test_metrics['fold'] = fold_idx+1
                fold_test_metrics.append(test_metrics)
                
                # Save fold test metrics to CSV
                fold_test_file = os.path.join(fold_output_dir, f"test_metrics_fold_{fold_idx+1}.csv")
                pd.DataFrame([test_metrics]).to_csv(fold_test_file)
                logger.info(f"Saved fold {fold_idx+1} test metrics to {fold_test_file}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx+1}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Aggregate results across folds
        logger.info("\n=== Cross-Validation Results ===")
        
        # Combine all fold results
        if fold_results:
            all_folds_df = pd.concat(fold_results, ignore_index=True)
            
            # Save combined results
            cv_results_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_cv_results.csv"
            all_folds_df.to_csv(cv_results_file)
            logger.info(f"Saved combined training results to {cv_results_file}")
        else:
            all_folds_df = pd.DataFrame()
            logger.warning("No valid fold results to combine")
        
        # Calculate average test metrics across folds
        if fold_test_metrics:
            test_metrics_df = pd.DataFrame(fold_test_metrics)
            
            # Calculate mean and std for each metric
            metric_columns = [col for col in test_metrics_df.columns if col != 'fold']
            summary_stats = {}
            
            for metric in metric_columns:
                mean_val = test_metrics_df[metric].mean()
                std_val = test_metrics_df[metric].std()
                summary_stats[f"{metric}_mean"] = mean_val
                summary_stats[f"{metric}_std"] = std_val
                logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Save aggregated test metrics
            cv_test_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_cv_test_metrics.csv"
            pd.DataFrame([summary_stats]).to_csv(cv_test_file)
            logger.info(f"Saved aggregated test metrics to {cv_test_file}")
            
            # Find the best performing fold
            if 'accuracy' in test_metrics_df.columns:
                best_fold_idx = test_metrics_df['accuracy'].idxmax()
            elif 'f1_score' in test_metrics_df.columns:
                best_fold_idx = test_metrics_df['f1_score'].idxmax()
            elif 'matthews_corrcoef' in test_metrics_df.columns:
                best_fold_idx = test_metrics_df['matthews_corrcoef'].idxmax()
            else:
                # Just take the last fold if no clear metric to select on
                best_fold_idx = len(fold_models) - 1
                
            # Get the best model
            if fold_models:
                best_model = fold_models[best_fold_idx]
                best_fold = test_metrics_df.iloc[best_fold_idx]['fold']
                logger.info(f"Best model from fold {best_fold} selected")
            else:
                best_model = None
                logger.warning("No models available from folds")
        else:
            logger.warning("No valid test metrics across folds")
            best_model = None if not fold_models else fold_models[-1]
        
        return all_folds_df, best_model
        
    except ImportError as e:
        logger.error(f"Failed to import cross-validation utilities: {e}")
        raise ImportError("Cross-validation requires split_utils.py with cross-validation functions")

def run_standard_pipeline(args, tokenizer):
    """
    Run the standard (non-cross-validation) training pipeline.
    This is the original workflow but extracted as a separate function.
    
    Args:
        args: Command-line arguments
        tokenizer: Tokenizer for the model
        
    Returns:
        results_df: DataFrame with training results
        model: Trained model
    """
    # Setup model with appropriate pruning method
    if args.pruning_method == "clustering":
        model, token_map, oov_lookup = setup_clustering_based_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            clustering_method=args.clustering_method
        )
    elif args.pruning_method == "frequency":
        model, token_map, oov_lookup = setup_frequency_based_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent
        )
    elif args.pruning_method == "frequency_oov":
        model, token_map, oov_lookup = setup_frequency_oov_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            num_clusters=args.num_clusters
        )
    elif args.pruning_method == "importance_oov":
        model, token_map, oov_lookup = setup_importance_oov_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            num_clusters=args.num_clusters,
            importance_type=args.importance_type
        )
    elif args.pruning_method == "importance":
        model, token_map, oov_lookup = setup_importance_based_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            importance_type=args.importance_type
        )
    elif args.pruning_method == "random":
        model, token_map, oov_lookup = setup_random_based_model(
            args.task, 
            args.model_name,
            prune_percent=args.prune_percent,
            random_seed=args.seed
        )
    elif args.pruning_method == "attention":
        model, token_map, oov_lookup = setup_attention_based_model(
            args.task, 
            args.model_name,
            attention_model=args.attention_model,
            prune_percent=args.prune_percent
        )
    elif args.pruning_method == "train_only":
        model, token_map, oov_lookup = setup_train_only_model(
            args.task, 
            args.model_name
        )
    elif args.pruning_method == "no_pruning":
        model, token_map, oov_lookup = setup_no_pruning_model(
            args.task, 
            args.model_name
        )
    
    # Always use custom splits for all methods including no_pruning
    logger.info("Creating custom train/validation/test splits")
    try:
        # Import custom splitting utilities
        from split_utils import prepare_custom_split_datasets, prepare_original_val_datasets
        
        # Use original validation set with train/test split instead of fully custom splits
        train_dataset, eval_dataset, test_dataset = prepare_original_val_datasets(
            task_name=args.task,
            tokenizer=tokenizer,
            train_ratio=0.9,  # Use 90% of original training data for training
            test_ratio=0.1,   # Use 10% of original training data for testing
            max_length=128,   # Same as in original prepare_datasets_with_mapping
            random_seed=args.seed
        )
        
        # For no_pruning, we'll use the datasets as-is (no remapping needed)
        if args.pruning_method == "no_pruning":
            logger.info("Using original validation set with train/test split without token remapping for no_pruning baseline")
            # Just make sure the label column is named correctly
            for dataset in [train_dataset, eval_dataset, test_dataset]:
                if dataset is not None and 'label' in dataset.column_names and 'labels' not in dataset.column_names:
                    dataset = dataset.rename_column('label', 'labels')
        else:
            # Apply token remapping to create datasets with reduced vocabulary
            logger.info("Applying vocabulary mapping to custom splits")
            
            # Function to remap tokens in a dataset
            def remap_tokens(dataset):
                # If no token_map (no pruning case), return the dataset unchanged
                if token_map is None:
                    logger.info("No token mapping applied (using full vocabulary)")
                    return dataset
                
                # Create a mapping dictionary for quick lookup
                id_map_dict = token_map.copy()
                if oov_lookup:
                    id_map_dict.update(oov_lookup)
                
                def map_example(example):
                    # Map input_ids using our token mapping
                    new_input_ids = []
                    for token_id in example['input_ids']:
                        if token_id in id_map_dict:
                            new_input_ids.append(id_map_dict[token_id])
                        else:
                            # Use UNK token (0) for OOV tokens
                            new_input_ids.append(0)
                    
                    example['input_ids'] = new_input_ids
                    return example
                
                # Apply mapping to each example
                remapped_dataset = dataset.map(map_example, desc="Remapping token IDs")
                
                # Rename 'label' to 'labels' if it exists (important for data collator compatibility)
                if 'label' in remapped_dataset.column_names and 'labels' not in remapped_dataset.column_names:
                    logger.info("Renaming 'label' column to 'labels' for compatibility with data collator")
                    remapped_dataset = remapped_dataset.rename_column('label', 'labels')
                
                return remapped_dataset
            
            # Apply remapping to all datasets for pruning methods
            train_dataset = remap_tokens(train_dataset)
            eval_dataset = remap_tokens(eval_dataset)
            test_dataset = remap_tokens(test_dataset)
        
        logger.info(f"Created custom splits with sizes: train={len(train_dataset)}, validation={len(eval_dataset)}, test={len(test_dataset)}")
        
    except ImportError as e:
        logger.error(f"Failed to import split_utils module: {e}")
        logger.error("Custom splits is now required. Please make sure split_utils.py is in your path.")
        raise ImportError("Could not import split_utils module which is required for custom splits")
    
    # Setup trainer
    if args.pruning_method == "no_pruning":
        # For no_pruning, use the standard tokenizer with the Trainer
        trainer, metrics_callback = setup_training(
            model, 
            train_dataset, 
            eval_dataset, 
            args.task, 
            args,
            tokenizer=tokenizer  # Pass tokenizer for standard DataCollatorWithPadding
        )
    else:
        # For pruning methods, use the custom collator
        trainer, metrics_callback = setup_training(
            model, 
            train_dataset, 
            eval_dataset, 
            args.task, 
            args
        )
    
    # Log initial vocabulary statistics
    logger.info(f"\n=== Vocabulary Statistics ===")
    logger.info(f"Original vocabulary size: {len(tokenizer.get_vocab())}")
    if token_map is None:
        logger.info("No tokens pruned (using full vocabulary)")
        logger.info(f"Vocabulary reduction: 0.00%")
    else:
        logger.info(f"Kept tokens: {len(token_map)}")
        if oov_lookup:
            logger.info(f"OOV clusters: {len(set(oov_lookup.values()))}")
        logger.info(f"Vocabulary reduction: {(1 - len(token_map)/len(tokenizer.get_vocab()))*100:.2f}%")
    
    # Log custom split ratios
    logger.info(f"Using original validation set with train/test split: train=90% of original train set, test=10% of original train set")
    
    # Train and evaluate
    logger.info(f"Starting training for {args.epochs} epochs")
    try:
        trainer.train()
        
        # Evaluate the model
        logger.info("Running final evaluation")
        eval_results = trainer.evaluate()
        for metric_name, value in eval_results.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Save training results
        results_df = create_results_dataframe(metrics_callback)
        
        # Save results to CSV
        if not results_df.empty:
            results_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_results.csv"
            results_df.to_csv(results_file)
            logger.info(f"Saved training results to {results_file}")
        
        # Analyze model size
        model_params = count_model_parameters(trainer.model)
        
        # Load original model for comparison
        logger.info("\nAnalyzing original model for comparison")
        original_model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        original_params = count_model_parameters(original_model)
        
        # Calculate reduction percentages
        logger.info("\n=== Parameter Reduction Statistics ===")
        for param_type in ['total', 'embedding', 'model_only']:
            reduction = (original_params[param_type] - model_params[param_type]) / original_params[param_type] * 100
            logger.info(f"{param_type.title()} parameter reduction: {reduction:.2f}%")
        
        # Log summary of results
        logger.info("\n=== Summary Results ===")
        logger.info(f"Task: {args.task}")
        logger.info(f"Pruning method: {args.pruning_method}")
        if args.pruning_method != "no_pruning":
            logger.info(f"Prune percent: {args.prune_percent}%")
            if args.pruning_method in ["frequency_oov", "importance_oov"]:
                logger.info(f"Num OOV clusters: {args.num_clusters}")
            if args.pruning_method == "importance_oov":
                logger.info(f"Importance type: {args.importance_type}")
            logger.info(f"Vocabulary reduction: {(1 - len(token_map)/len(tokenizer.get_vocab()))*100:.2f}%")
        else:
            logger.info("No vocabulary pruning applied (baseline method)")
            logger.info("Vocabulary reduction: 0.00%")
        
        # Extract final metrics
        final_metrics = {}
        for metric_name, value in eval_results.items():
            if metric_name.startswith("eval_"):
                metric_short_name = metric_name[5:]  # Remove 'eval_' prefix
                final_metrics[metric_short_name] = value
        
        # Log performance metrics
        for metric_name, value in final_metrics.items():
            logger.info(f"Final {metric_name}: {value:.4f}")
        
        # After training, generate predictions on test set
        logger.info("\n=== Test Set Prediction Generation ===")
        try:
            # Generate test predictions using the function from training.py
            if args.pruning_method == "no_pruning":
                generate_test_predictions(
                    trainer.model,
                    test_dataset, 
                    trainer.data_collator,
                    args,
                    tokenizer=tokenizer,  # Pass tokenizer for no_pruning
                    trainer=trainer       # Pass the trainer for consistent processing
                )
            else:
                generate_test_predictions(
                    trainer.model,
                    test_dataset, 
                    trainer.data_collator,
                    args,
                    trainer=trainer       # Pass the trainer for consistent processing
                )
            
            # Test metrics can now be calculated directly with a helper function
            test_metrics = evaluate_test_set(
                trainer.model,
                test_dataset, 
                args.task, 
                trainer.data_collator, 
                args.batch_size,
                args.pruning_method,
                tokenizer,
                trainer  # Pass the trainer for consistent processing
            )
            
            # Save test metrics to CSV
            test_metrics_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_test_metrics.csv"
            pd.DataFrame([test_metrics]).to_csv(test_metrics_file)
            logger.info(f"Saved test metrics to {test_metrics_file}")
            
        except Exception as e:
            logger.error(f"Error generating test predictions: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return results_df, trainer.model
    
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error occurred: {e}")
            logger.info("Attempting to free memory and continue with CPU...")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move model to CPU for evaluation
            model = model.cpu()
            
            # Create empty results dataframe
            results_df = pd.DataFrame()
            
            return results_df, model
        else:
            raise

def run_pipeline(args):
    """
    Execute the full training pipeline with vocabulary pruning.
    
    Args:
        args: Command-line arguments
        
    Returns:
        results_df: DataFrame with training results
        model: Trained model
    """
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file
    log_filename = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}.log"
    if args.pruning_method == "clustering":
        log_filename = f"{args.output_dir}/{args.task}_clustering_prune{args.prune_percent}_{args.clustering_method}.log"
    elif args.pruning_method == "frequency_oov":
        log_filename = f"{args.output_dir}/{args.task}_frequency_oov_prune{args.prune_percent}_clusters{args.num_clusters}.log"
    elif args.pruning_method == "importance_oov":
        log_filename = f"{args.output_dir}/{args.task}_importance_oov_prune{args.prune_percent}_clusters{args.num_clusters}_type{args.importance_type}.log"
    elif args.pruning_method == "importance":
        log_filename = f"{args.output_dir}/{args.task}_importance_prune{args.prune_percent}_type{args.importance_type}.log"
    elif args.pruning_method == "random":
        log_filename = f"{args.output_dir}/{args.task}_random_prune{args.prune_percent}.log"
    elif args.pruning_method == "attention":
        attention_model_suffix = "_finetuned" if args.attention_model else ""
        log_filename = f"{args.output_dir}/{args.task}_attention_prune{args.prune_percent}{attention_model_suffix}.log"
    elif args.pruning_method == "train_only":
        log_filename = f"{args.output_dir}/{args.task}_train_only.log"
    elif args.pruning_method == "no_pruning":
        log_filename = f"{args.output_dir}/{args.task}_no_pruning.log"
    
    # Add CV info to log filename if using cross-validation
    if args.cross_validation:
        log_filename = log_filename.replace(".log", f"_cv{args.n_folds}.log")
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info(f"Running with arguments: {args}")
    
    # Initialize tokenizer for the original model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.cross_validation:
        logger.info(f"Using {args.n_folds}-fold cross-validation")
        return run_cross_validation_pipeline(args, tokenizer)
    else:
        return run_standard_pipeline(args, tokenizer)

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Run pipeline
    try:
        results_df, model = run_pipeline(args)
        logger.info("Pipeline completed successfully")
        return results_df, model
    except Exception as e:
        logger.exception(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 