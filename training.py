"""
Training functions for vocabulary pruning.
"""

import logging
from functools import partial
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import traceback

from metrics import MetricsCallback, compute_metrics
from data import ReducedVocabDataCollator, HybridCollator

# Configure logging
logger = logging.getLogger(__name__)

def setup_training(model, train_dataset, eval_dataset, task_name, args, tokenizer=None):
    """
    Set up training and evaluation for a reduced vocabulary model.
    
    Args:
        model: Model with reduced vocabulary
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        task_name: Name of the GLUE task
        args: Command-line arguments with training hyperparameters
        tokenizer: Tokenizer object (required for no_pruning case)
        
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
    if args.pruning_method == "no_pruning" and tokenizer is not None:
        # For no_pruning, use the standard DataCollatorWithPadding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    elif args.pruning_method in ["frequency_oov", "importance_oov"]:
        # For OOV methods using TokenMappingWrapper, use ReducedVocabDataCollator
        # The token remapping is now handled in the model's forward pass
        logger.info("Using ReducedVocabDataCollator with TokenMappingWrapper")
        data_collator = ReducedVocabDataCollator(pad_token_id=0)
    elif args.pruning_method in ["hybrid", "importance"]:
        data_collator = HybridCollator(pad_token_id=0, unk_token_id=0)
    else:
        data_collator = ReducedVocabDataCollator(pad_token_id=0)
    
    # Setup metrics callback
    metrics_callback = MetricsCallback()
    
    # Setup trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": partial(compute_metrics, task_name=task_name),
    }
    
    # Pass tokenizer to trainer for no_pruning case
    if args.pruning_method == "no_pruning" and tokenizer is not None:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = Trainer(**trainer_kwargs)
    
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

def generate_test_predictions(model, test_dataset, data_collator, args, tokenizer=None, trainer=None):
    """
    Generate predictions on the test set.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        data_collator: Data collator for batching
        args: Command line arguments
        tokenizer: Tokenizer for no_pruning case
        trainer: The trainer object (if available) for consistent processing
        
    Returns:
        predictions: Model predictions on test set
    """
    if test_dataset is None:
        logger.info("No test dataset available, skipping test predictions")
        return None
    
    logger.info("\n=== Test Set Prediction Generation ===")
    try:
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
            logger.info("Using manual prediction generation")
            predictions = []
            
            # Create dataloader for test set
            dataloader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=args.batch_size,
                collate_fn=data_collator
            )
            
            # Move model to appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            # Process batches
            for batch in tqdm(dataloader, desc="Generating predictions"):
                # Filter out non-tensor values and labels
                batch_inputs = {k: v.to(device) for k, v in batch.items() 
                               if k != 'labels' and k != 'prediction_only' and hasattr(v, 'to')}
                
                with torch.no_grad():
                    outputs = model(**batch_inputs)
                logits = outputs.logits
                
                # Convert logits to predictions
                if len(logits.shape) > 1 and logits.shape[1] > 1:
                    # For classification tasks
                    batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    # For regression tasks
                    batch_preds = logits.cpu().squeeze().numpy()
                
                predictions.append(batch_preds)
            
            # Combine predictions into single array
            if isinstance(predictions[0], np.ndarray) and len(predictions[0].shape) > 0:
                predictions = np.concatenate(predictions, axis=0)
            else:
                predictions = np.array(predictions)
        
        # Save predictions to file
        output_test_file = f"{args.output_dir}/{args.task}_{args.pruning_method}_prune{args.prune_percent}_predictions.txt"
        np.savetxt(output_test_file, predictions, fmt='%d' if len(predictions.shape) == 1 else '%.6f')
        logger.info(f"Saved test predictions to {output_test_file}")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error generating test predictions: {e}")
        logger.error(traceback.format_exc())
        return None