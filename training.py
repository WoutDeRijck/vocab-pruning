"""
Training functions for vocabulary pruning.
"""

import logging
from functools import partial
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import TrainingArguments, Trainer

from metrics import MetricsCallback, compute_metrics
from data import ReducedVocabDataCollator, HybridCollator

# Configure logging
logger = logging.getLogger(__name__)

def setup_training(model, train_dataset, eval_dataset, task_name, args):
    """
    Set up training and evaluation for a reduced vocabulary model.
    
    Args:
        model: Model with reduced vocabulary
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        task_name: Name of the GLUE task
        args: Command-line arguments with training hyperparameters
        
    Returns:
        trainer: Configured Trainer object
        metrics_callback: Callback for tracking metrics
    """
    # Training hyperparameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available(),  # Use bfloat16 if available
        bf16_full_eval=torch.cuda.is_available(),
        push_to_hub=False,
        seed=args.seed,
        torch_compile=False,  # Disable torch compilation to avoid CUDA errors
    )
    
    # Create data collator based on pruning method
    if args.pruning_method in ["hybrid", "importance"]:
        data_collator = HybridCollator(pad_token_id=0, unk_token_id=0)
    else:
        data_collator = ReducedVocabDataCollator(pad_token_id=0)
    
    # Setup metrics callback
    metrics_callback = MetricsCallback()
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, task_name=task_name),
    )
    
    # Add callback
    trainer.add_callback(metrics_callback)
    
    return trainer, metrics_callback

def save_training_results(metrics_callback, args, task_name, output_dir):
    """
    Save training results to a CSV file.
    
    Args:
        metrics_callback: Callback containing training metrics
        args: Command line arguments
        task_name: GLUE task name
        output_dir: Output directory for results
        
    Returns:
        results_df: DataFrame with metrics
    """
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
            
            # Save results
            results_file = f"{output_dir}/{task_name}_{args.pruning_method}_prune{args.prune_percent}_results.csv"
            results_df.to_csv(results_file)
            logger.info(f"Saved training results to {results_file}")
            
            return results_df
        else:
            logger.warning("No training history available, skipping results DataFrame creation")
            return pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error creating results DataFrame: {e}")
        logger.warning("Continuing without saving training history")
        return pd.DataFrame()

def generate_test_predictions(model, test_dataset, data_collator, args):
    """
    Generate predictions on the test set.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        data_collator: Data collator for batching
        args: Command line arguments
        
    Returns:
        predictions: Model predictions on test set
    """
    if test_dataset is None:
        logger.info("No test dataset available, skipping test predictions")
        return None
    
    logger.info("\n=== Test Set Prediction Generation ===")
    try:
        # Generate predictions without computing loss
        logger.info("Generating predictions on test set")
        with torch.no_grad():
            device = next(model.parameters()).device
            predictions = []
            
            # Process in batches
            dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size,
                collate_fn=data_collator
            )
            
            for batch in tqdm(dataloader, desc="Predicting"):
                # Filter out non-tensor values and labels
                batch = {k: v.to(device) for k, v in batch.items() 
                        if k != 'labels' and k != 'prediction_only' and hasattr(v, 'to')}
                with torch.no_grad():
                    outputs = model(**batch)
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
        
        # Save predictions for submission
        output_test_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_predictions.txt"
        np.savetxt(output_test_file, predictions, fmt='%d' if len(predictions.shape) == 1 else '%.6f')
        logger.info(f"Saved test predictions to {output_test_file}")
        
        return predictions
    except Exception as e:
        logger.error(f"Error generating test predictions: {e}")
        return None 