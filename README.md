# Vocabulary Pruning

This directory contains an implementation for vocabulary pruning techniques to reduce model size while maintaining performance on GLUE benchmark tasks.

## Overview

The implementation combines several vocabulary pruning techniques:

1. **Clustering-based Pruning**: Groups similar token embeddings using clustering algorithms and keeps representative tokens from each cluster
2. **Frequency-based Pruning**: Keeps the most frequently occurring tokens in the dataset and removes rare tokens
3. **Hybrid Pruning (Frequency+OOV)**: Combines frequency-based pruning with clustering of removed tokens for better OOV token handling
4. **Word Importance Pruning (with OOV)**: Uses TF-IDF scores to determine token importance, with clustering for OOV token handling
5. **Word Importance Pruning (without OOV)**: Uses TF-IDF scores for pruning but maps OOV tokens to UNK
6. **Random Selection**: Tokens are pruned randomly without consideration for importance, serving as a baseline approach
7. **Attention-based Pruning**: Uses attention patterns from a fine-tuned model to extract contextual token importance
8. **Train-only Pruning**: Simply keeps tokens that appear in the training set without any additional pruning
9. **No-pruning Baseline**: Keeps the full original vocabulary without any changes for baseline comparison

## Usage

The main script supports all pruning methods through a single interface:

```bash
python main.py --task sst2 --pruning_method clustering --prune_percent 20 --epochs 3
```

We also provide task-specific example scripts in the `examples/` directory:

```bash
# Example for running a task-specific script
python examples/run_clustering_pruning.py --task mrpc --prune_percent 20
```

### Common Arguments

- `--task`: GLUE task name (default: "sst2")
- `--model_name`: Pretrained model name (default: "answerdotai/ModernBERT-base")
- `--pruning_method`: Pruning method ["clustering", "frequency", "frequency_oov", "importance_oov", "importance", "random", "attention", "train_only", "no_pruning"] (default: "clustering")
- `--prune_percent`: Percentage of vocabulary to prune (default: 20)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 8e-5)
- `--weight_decay`: Weight decay (default: 8e-6)
- `--batch_size`: Training batch size (default: 32)
- `--output_dir`: Output directory for model checkpoints and logs (default: "./modular_model_output")
- `--train_only`: Use only the training set for vocabulary extraction (flag)
- `--seed`: Random seed (default: 42)

### Method-specific Arguments

#### Clustering-based Pruning
- `--clustering_method`: Clustering algorithm ["agglomerative", "kmeans"] (default: "agglomerative")

#### Hybrid Pruning
- `--num_clusters`: Number of clusters for OOV token mapping (default: 50)

#### Importance-based Pruning
- `--importance_type`: Word importance setting [0=off, 1=no norm, 2=L1 norm, 3=L2 norm] (default: 3)
- `--num_clusters`: Number of clusters for OOV token mapping (default: 50) (only for importance_oov method)

#### Attention-based Pruning
- `--attention_model`: Path to a fine-tuned model to use for attention-based importance calculation (default: uses the base model)

### Data Split Arguments

- `--cross_validation`: Use cross-validation instead of fixed train/val/test split (flag)
- `--n_folds`: Number of folds for cross-validation (default: 5)
- `--train_ratio`: Ratio of data to use for training (default: 0.8)
- `--validation_ratio`: Ratio of data to use for validation (default: 0.1)
- `--test_ratio`: Ratio of data to use for testing (default: 0.1)

## Example Scripts

The `examples/` directory contains scripts for running each pruning method on specific GLUE tasks. These scripts make it easy to run experiments with task-optimized defaults:

- `run_clustering_pruning.py`: Clustering-based pruning
- `run_frequency_pruning.py`: Basic frequency-based pruning
- `run_frequency_oov_pruning.py`: Frequency-based pruning with OOV clustering
- `run_importance_oov_pruning.py`: Word importance pruning with OOV clustering
- `run_importance_pruning.py`: Word importance pruning without OOV clustering
- `run_random_pruning.py`: Random token selection
- `run_attention_pruning.py`: Attention-based pruning
- `run_train_only_pruning.py`: Train-only vocabulary pruning
- `run_no_pruning.py`: No-pruning baseline

Each script supports the `--tasks` parameter to specify which GLUE tasks to run, and contains task-specific defaults for parameters like learning rate, batch size, and pruning percentage. You can override these defaults using command-line arguments.

### Summary Functionality

All example scripts now include a summary functionality that collects and displays results across all GLUE tasks at the end of execution:

- Test metrics (accuracy, F1, etc.) for each task
- Parameter reduction statistics (vocabulary, embedding, total)
- Comprehensive results saved to a CSV file with timestamp

This makes it easy to compare the effectiveness of different pruning methods across all GLUE tasks.

## Examples

### Clustering-based Pruning
```bash
python main.py --task mrpc --pruning_method clustering --prune_percent 25 --clustering_method kmeans --epochs 5
# Or using example script:
python examples/run_clustering_pruning.py --task mrpc --prune_percent 25 --clustering_method kmeans
```

### Frequency-based Pruning
```bash
python main.py --task sst2 --pruning_method frequency --prune_percent 30 --epochs 3
# Or using example script:
python examples/run_frequency_pruning.py --task sst2 --prune_percent 30
```

### Hybrid Pruning (Frequency + OOV)
```bash
python main.py --task cola --pruning_method frequency_oov --prune_percent 20 --num_clusters 50 --epochs 10
# Or using example script:
python examples/run_frequency_oov_pruning.py --task cola --prune_percent 20 --num_clusters 50
```

### Word Importance Pruning (with OOV)
```bash
python main.py --task qnli --pruning_method importance_oov --prune_percent 15 --num_clusters 100 --importance_type 3 --epochs 3
# Or using example script:
python examples/run_importance_oov_pruning.py --task qnli --prune_percent 15 --num_clusters 100 --importance_type 3
```

### Word Importance Pruning (without OOV)
```bash
python main.py --task sst2 --pruning_method importance --prune_percent 20 --importance_type 3 --epochs 3
# Or using example script:
python examples/run_importance_pruning.py --task sst2 --prune_percent 20 --importance_type 3
```

### Random Selection Pruning
```bash
python main.py --task mrpc --pruning_method random --prune_percent 20 --epochs 3
# Or using example script:
python examples/run_random_pruning.py --task mrpc --prune_percent 20
```

### Attention-based Pruning
```bash
# Using the base model for attention calculation
python main.py --task mrpc --pruning_method attention --prune_percent 20 --epochs 3

# Using a fine-tuned model for attention calculation
python main.py --task mrpc --pruning_method attention --prune_percent 20 --attention_model path/to/finetuned/model --epochs 3

# Or using example script:
python examples/run_attention_pruning.py --task mrpc --prune_percent 20 --finetuned_model_path path/to/finetuned/model
```

### Train-only Pruning
```bash
python main.py --task sst2 --pruning_method train_only --epochs 3
# Or using example script:
python examples/run_train_only_pruning.py --tasks sst2 mrpc
```

### No-pruning Baseline
```bash
python main.py --task sst2 --pruning_method no_pruning --epochs 3
# Or using example script:
python examples/run_no_pruning.py --tasks sst2 mrpc
```

### Using Task-Specific Defaults

Each example script includes task-specific defaults for parameters like learning rate, batch size, epochs, and pruning percentage. You can run scripts with these defaults:

```bash
# Run with defaults for SST-2 task
python examples/run_random_pruning.py --task sst2

# Run with defaults for MRPC task but override pruning percentage
python examples/run_importance_pruning.py --task mrpc --prune_percent 30

# Run with defaults for CoLA task but override importance type
python examples/run_importance_pruning.py --task cola --importance_type 2
```

### Running on Multiple Tasks

Many of the example scripts support running on multiple GLUE tasks in sequence. For these scripts, you can use the `--tasks` argument to specify which tasks to run:

```bash
# Run on multiple specific tasks
python examples/run_frequency_oov_pruning.py --tasks sst2 cola mrpc

# Run on all GLUE tasks
python examples/run_importance_oov_pruning.py --tasks cola mnli mrpc qnli qqp rte sst2 stsb wnli

# Run on multiple tasks with a shared pruning percentage
python examples/run_clustering_pruning.py --tasks mrpc rte sst2 --prune_percent 25
```

Each task will be run sequentially using its task-specific default parameters, which can be overridden with command-line arguments. Results will be saved in task-specific subdirectories within the output directory, and a summary of results across all tasks will be displayed at the end.

Scripts that support multiple tasks with the summary functionality include:
- `run_clustering_pruning.py`
- `run_frequency_pruning.py`
- `run_frequency_oov_pruning.py`
- `run_importance_oov_pruning.py`
- `run_random_pruning.py`
- `run_train_only_pruning.py`
- `run_no_pruning.py`

### Using Cross-Validation
```bash
python main.py --task sst2 --pruning_method clustering --prune_percent 20 --cross_validation --n_folds 5
```

### Custom Train/Validation/Test Split
```bash
python main.py --task cola --pruning_method frequency --prune_percent 25 --train_ratio 0.7 --validation_ratio 0.15 --test_ratio 0.15
```

## Output

The script produces:
- Trained model checkpoints in the specified output directory
- Logs with detailed information about pruning and training
- CSV files with training metrics
- Test predictions for submission to GLUE benchmark (if test set is available)
- Summary of results across all tasks (when using example scripts with multiple tasks)

## How It Works

1. **Setup**: Loads the specified GLUE task and pretrained model
2. **Vocabulary Analysis**: Analyzes the task-specific vocabulary requirements
3. **Pruning**: Applies the selected pruning method to reduce vocabulary size
4. **Model Adaptation**: Replaces the embedding layer with a smaller one
5. **Training**: Fine-tunes the model on the task
6. **Evaluation**: Evaluates performance using official GLUE metrics
7. **Analysis**: Reports vocabulary reduction and parameter savings
8. **Summary**: Generates a comprehensive summary of results across tasks (when using example scripts)

## Troubleshooting

If you encounter CUDA out-of-memory errors:
- Reduce batch size
- Use a smaller model
- Lower the clustering complexity or number of clusters

For issues with word importance methods, ensure nltk is installed:
```
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

## Visualization and Results

The repository includes a visualization script that generates plots comparing all pruning methods:

```bash
python visualize_results.py
```

This script produces several visualizations:

1. **Performance Comparison**: Bar charts comparing test performance across all methods
2. **Performance Drop**: Charts showing the performance drop from the baseline for each method
3. **Parameter Reduction**: Bar charts showing total, embedding, and vocabulary parameter reduction
4. **Dataset Size Impact**: Scatter plots showing the relationship between dataset size and performance
5. **Method-specific Comparisons**: Direct comparisons between regular and OOV variants of methods
6. **Performance vs. Parameter Reduction**: Scatter plots showing the tradeoff between performance and model size

Results are summarized in three CSV files:

1. **SUMMARY.csv**: Raw performance metrics for all methods (train and test)
2. **SUMMARY_WITH_SIZES.csv**: Performance metrics and relative drop for each method, organized by dataset size
3. **PARAM_REDUCTION_SUMMARY.csv**: Parameter reduction percentages (total, embedding, vocabulary) across methods

These files and visualizations make it easy to compare:
- Performance across different pruning techniques
- Tradeoffs between model size and accuracy
- Method effectiveness based on dataset properties

Each visualization includes all implemented pruning methods:
- NO PRUNING (baseline)
- TRAIN TOKENS ONLY
- RANDOM
- CLUSTERING
- FREQUENCY
- ATTENTION
- IMPORTANCE
- FREQUENCY_OOV
- IMPORTANCE_OOV
