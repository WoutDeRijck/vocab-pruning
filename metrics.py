"""
Metrics and callbacks for vocabulary pruning evaluation.
"""

import logging
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from transformers import TrainerCallback
from evaluate import load

# Configure logging
logger = logging.getLogger(__name__)

class MetricsCallback(TrainerCallback):
    """Callback to track training and evaluation metrics."""
    
    def __init__(self):
        """Initialize metrics collection."""
        self.training_history = {
            "train": {
                "loss": [],
                "learning_rate": [],
                "epoch": [],
            },
            "eval": {
                "loss": [],
                "epoch": [],
            }
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Collect metrics when logged."""
        if logs is None:
            return
        
        logs = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        
        # Determine if train or eval metrics
        is_eval = any(k.startswith("eval_") for k in logs.keys())
        
        if is_eval:
            # Extract and store eval metrics
            for key, value in logs.items():
                if key.startswith("eval_"):
                    metric_name = key[5:]  # Remove 'eval_' prefix
                    if metric_name not in self.training_history["eval"]:
                        self.training_history["eval"][metric_name] = []
                    self.training_history["eval"][metric_name].append(value)
                elif key == "epoch":
                    self.training_history["eval"]["epoch"].append(value)
        else:
            # Extract and store training metrics
            for key, value in logs.items():
                if key in self.training_history["train"]:
                    self.training_history["train"][key].append(value)

def compute_metrics(eval_pred, task_name):
    """
    Compute metrics for model evaluation using official GLUE metrics.
    
    Args:
        eval_pred: Tuple of predictions and labels
        task_name: Name of the GLUE task
        
    Returns:
        results: Dictionary of metric results
    """
    predictions, labels = eval_pred
    
    # Handle regression task (e.g., STS-B)
    if len(predictions.shape) == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze()
    # Handle classification task
    elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Load the official GLUE metric for this task
    metric = load('glue', task_name)
    
    # Compute metrics
    # Only compute if we have valid labels (not -1)
    valid_indices = labels != -1
    if valid_indices.any():
        results = metric.compute(predictions=predictions[valid_indices], 
                               references=labels[valid_indices])
    else:
        logger.info("No valid labels found (test set?), skipping metric computation")
        results = {}
    
    return results

def count_model_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        counts: Dictionary with parameter counts
    """
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Try to identify the embedding layer(s)
    embedding_params = 0
    
    # For ModernBERT model structure
    if hasattr(model, 'model') and hasattr(model.model, 'embeddings'):
        if hasattr(model.model.embeddings, 'tok_embeddings'):
            embedding_params += model.model.embeddings.tok_embeddings.weight.numel()
        # Check for other embedding types like position embeddings
        for name, param in model.model.embeddings.named_parameters():
            if name != 'tok_embeddings.weight' and 'embed' in name:
                embedding_params += param.numel()
    
    # For standard BERT model structure
    elif hasattr(model, 'embeddings'):
        if hasattr(model.embeddings, 'word_embeddings'):
            embedding_params += model.embeddings.word_embeddings.weight.numel()
        # Check for other embedding types
        for name, param in model.embeddings.named_parameters():
            if name != 'word_embeddings.weight' and 'embed' in name:
                embedding_params += param.numel()
    
    # Calculate model-only parameters (excluding embeddings)
    model_only_params = total_params - embedding_params
    
    return {
        'total': total_params,
        'embedding': embedding_params,
        'model_only': model_only_params
    } 