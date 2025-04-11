# Unified Vocabulary Pruning

This directory contains a unified implementation for vocabulary pruning techniques to reduce model size while maintaining performance on GLUE benchmark tasks.

## Overview

The unified implementation combines four different vocabulary pruning techniques:

1. **Clustering-based Pruning**: Groups similar token embeddings using clustering algorithms and keeps representative tokens from each cluster
2. **Frequency-based Pruning**: Keeps the most frequently occurring tokens in the dataset and removes rare tokens
3. **Hybrid Pruning**: Combines frequency-based pruning with clustering of removed tokens for better OOV token handling
4. **Word Importance Pruning**: Uses TF-IDF scores to determine token importance, prioritizing tokens that carry more semantic information

## Installation

```
pip install -r requirements.txt
```

Requirements:
- torch
- transformers
- datasets
- scikit-learn
- pandas
- numpy
- tqdm
- evaluate
- nltk (optional, for word importance methods)

## Usage

The unified script supports all pruning methods through a single interface:

```bash
python unified_pruning.py --task sst2 --pruning_method clustering --prune_percent 20 --epochs 3
```

### Common Arguments

- `--task`: GLUE task name (default: "sst2")
- `--model_name`: Pretrained model name (default: "answerdotai/ModernBERT-base")
- `--pruning_method`: Pruning method ["clustering", "frequency", "hybrid", "importance"] (default: "clustering")
- `--prune_percent`: Percentage of vocabulary to prune (default: 20)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 8e-5)
- `--batch_size`: Training batch size (default: 32)
- `--output_dir`: Output directory for model checkpoints and logs (default: "./unified_model_output")
- `--seed`: Random seed (default: 42)

### Method-specific Arguments

#### Clustering-based Pruning
- `--clustering_method`: Clustering algorithm ["agglomerative", "kmeans"] (default: "agglomerative")

#### Hybrid Pruning
- `--num_clusters`: Number of clusters for OOV token mapping (default: 50)

#### Importance-based Pruning
- `--importance_type`: Word importance setting [0=off, 1=no norm, 2=L1 norm, 3=L2 norm] (default: 3)
- `--num_clusters`: Number of clusters for OOV token mapping (default: 50)

## Examples

### Clustering-based Pruning
```bash
python unified_pruning.py --task mrpc --pruning_method clustering --prune_percent 25 --clustering_method kmeans --epochs 5
```

### Frequency-based Pruning
```bash
python unified_pruning.py --task sst2 --pruning_method frequency --prune_percent 30 --epochs 3
```

### Hybrid Pruning
```bash
python unified_pruning.py --task cola --pruning_method hybrid --prune_percent 20 --num_clusters 50 --epochs 10
```

### Word Importance Pruning
```bash
python unified_pruning.py --task qnli --pruning_method importance --prune_percent 15 --num_clusters 100 --importance_type 3 --epochs 3
```

## Output

The script produces:
- Trained model checkpoints in the specified output directory
- Logs with detailed information about pruning and training
- CSV files with training metrics
- Test predictions for submission to GLUE benchmark (if test set is available)

## How It Works

1. **Setup**: Loads the specified GLUE task and pretrained model
2. **Vocabulary Analysis**: Analyzes the task-specific vocabulary requirements
3. **Pruning**: Applies the selected pruning method to reduce vocabulary size
4. **Model Adaptation**: Replaces the embedding layer with a smaller one
5. **Training**: Fine-tunes the model on the task
6. **Evaluation**: Evaluates performance using official GLUE metrics
7. **Analysis**: Reports vocabulary reduction and parameter savings

## Extending the Code

To add a new pruning method:
1. Implement the vocabulary extraction function
2. Implement the pruning algorithm
3. Add a model setup function
4. Update the argument parser and main pipeline

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
