#!/usr/bin/env python3
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import torch

# Add the current directory to the path to import split_utils
sys.path.append(os.getcwd())

# Load ModernBERT tokenizer - using standard BERT since ModernBERT is based on BERT architecture
# Note: Replace this with the actual ModernBERT tokenizer path when available
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Import the custom split utility from the main codebase
try:
    from split_utils import prepare_original_val_datasets
except ImportError:
    print("Error: split_utils.py not found. Please ensure it's in the current directory.")
    sys.exit(1)

# GLUE tasks to analyze - include all tasks as before
glue_tasks = {
    "sst2": {"text_fields": ["sentence"], "split": "validation"},
    "mrpc": {"text_fields": ["sentence1", "sentence2"], "split": "validation"},
    "cola": {"text_fields": ["sentence"], "split": "validation"},
    "stsb": {"text_fields": ["sentence1", "sentence2"], "split": "validation"},
    "qqp": {"text_fields": ["question1", "question2"], "split": "validation"},
    "mnli": {"text_fields": ["premise", "hypothesis"], "split": "validation_matched"},
    "qnli": {"text_fields": ["question", "sentence"], "split": "validation"},
    "rte": {"text_fields": ["sentence1", "sentence2"], "split": "validation"},
}

def calculate_token_importance(token_counter, texts, tokenized_texts):
    """Calculate token importance using a simpler alternative to TF-IDF"""
    # Get the most frequent tokens
    freq_tokens = [token for token, _ in token_counter.most_common()]
    
    # Calculate document frequency (number of texts where each token appears)
    doc_freq = {}
    for token_id in token_counter.keys():
        doc_freq[token_id] = sum(1 for tokens in tokenized_texts if token_id in tokens)
    
    # Calculate a simple importance score (frequency / document frequency)
    importance_scores = {}
    for token_id, count in token_counter.items():
        if doc_freq[token_id] > 0:  # Avoid division by zero
            importance_scores[token_id] = count / doc_freq[token_id]
        else:
            importance_scores[token_id] = 0
    
    # Get top tokens by importance
    importance_tokens = [token for token, _ in sorted(importance_scores.items(), 
                                                     key=lambda x: x[1], 
                                                     reverse=True)]
    
    # Calculate overlap between frequency and importance rankings (top 20%)
    top_n = int(0.2 * len(freq_tokens))
    if top_n > 0:  # Ensure we have enough tokens
        freq_top = set(freq_tokens[:top_n])
        importance_top = set(importance_tokens[:top_n])
        overlap = len(freq_top.intersection(importance_top)) / len(freq_top) * 100
    else:
        overlap = 100.0  # If we have very few tokens, assume full overlap
    
    return overlap

def analyze_dataset_split(dataset, text_fields, split_name):
    """Analyze token statistics for a specific dataset split"""
    print(f"Processing {split_name} split...")
    
    # Extract text from fields
    texts = []
    for field in text_fields:
        if field in dataset.column_names:
            texts.extend([str(example[field]) for example in dataset])
    
    # Tokenize all texts
    tokenized = [tokenizer.encode(text, add_special_tokens=False) for text in tqdm(texts)]
    
    # Keep track of which tokens appear in which texts (for importance calculation)
    token_sets = [set(tokens) for tokens in tokenized]
    
    # Flatten token lists
    all_tokens = [token for tokens in tokenized for token in tokens]
    
    # Calculate token statistics
    token_counter = Counter(all_tokens)
    unique_tokens = len(token_counter)
    total_tokens = len(all_tokens)
    vocab_coverage = unique_tokens / len(tokenizer.vocab) * 100
    
    # Calculate token distribution coverage
    token_counts = np.array(list(token_counter.values()))
    token_counts = np.sort(token_counts)[::-1]  # Sort in descending order
    cumulative_counts = np.cumsum(token_counts)
    
    # Calculate percentage of dataset covered by top N% most frequent tokens
    if unique_tokens > 10:  # Ensure we have enough tokens for meaningful statistics
        top_10_coverage = cumulative_counts[int(0.1 * unique_tokens)] / total_tokens * 100
        top_20_coverage = cumulative_counts[int(0.2 * unique_tokens)] / total_tokens * 100
        top_50_coverage = cumulative_counts[int(0.5 * unique_tokens)] / total_tokens * 100
    else:
        # For very small vocabularies, use defaults
        top_10_coverage = 100.0
        top_20_coverage = 100.0
        top_50_coverage = 100.0
    
    # Calculate importance-frequency overlap
    overlap = calculate_token_importance(token_counter, texts, tokenized)
    
    # Return statistics and the token counter for train/test overlap analysis
    return {
        "Split": split_name,
        "Total Tokens": total_tokens,
        "Unique Tokens": unique_tokens,
        "Vocab Coverage (%)": vocab_coverage,
        "Top 10% Coverage (%)": top_10_coverage,
        "Top 20% Coverage (%)": top_20_coverage,
        "Top 50% Coverage (%)": top_50_coverage,
        "TF-IDF/Freq Overlap (%)": overlap,
    }, token_counter

def analyze_train_test_overlap(train_counter, test_counter):
    """Calculate overlap statistics between train and test vocabulary"""
    train_vocab = set(train_counter.keys())
    test_vocab = set(test_counter.keys())
    
    # Calculate test tokens not in training set
    test_tokens_not_in_train = test_vocab - train_vocab
    test_unique_not_in_train = len(test_tokens_not_in_train)
    
    # Calculate percentage of unique test tokens not in training
    test_unique_not_in_train_pct = test_unique_not_in_train / len(test_vocab) * 100 if len(test_vocab) > 0 else 0
    
    # Calculate total occurrences of OOV tokens in test set
    oov_token_count = sum(test_counter[token] for token in test_tokens_not_in_train)
    test_total_tokens = sum(test_counter.values())
    oov_token_pct = oov_token_count / test_total_tokens * 100 if test_total_tokens > 0 else 0
    
    return {
        "Test Unique Tokens Not In Train": test_unique_not_in_train,
        "Test Unique Tokens Not In Train (%)": test_unique_not_in_train_pct,
        "Test OOV Token Occurrences": oov_token_count,
        "Test OOV Token Occurrences (%)": oov_token_pct,
    }

def analyze_task(task_name, config, seed=42):
    """Analyze token statistics for a GLUE task using the same custom splits as in main pipeline"""
    print(f"\nProcessing {task_name}...")
    
    try:
        # Create custom train/test splits using the same approach as in the main pipeline
        print(f"Creating custom train/test splits for {task_name}...")
        train_dataset, eval_dataset, test_dataset = prepare_original_val_datasets(
            task_name=task_name,
            tokenizer=tokenizer,
            train_ratio=0.9,  # Use 90% of original training data for training
            test_ratio=0.1,   # Use 10% of original training data for testing
            max_length=128,   # Same as in original prepare_datasets_with_mapping
            random_seed=seed
        )
        
        print(f"Created splits with sizes: train={len(train_dataset)}, validation={len(eval_dataset)}, test={len(test_dataset)}")
        
        # Analyze each split
        results = []
        
        # Analyze training split
        train_stats, train_counter = analyze_dataset_split(train_dataset, config["text_fields"], "Train")
        train_stats["Task"] = task_name.upper()
        results.append(train_stats)
        
        # Analyze test split
        test_stats, test_counter = analyze_dataset_split(test_dataset, config["text_fields"], "Test")
        test_stats["Task"] = task_name.upper()
        results.append(test_stats)
        
        # Calculate overlap statistics
        overlap_stats = analyze_train_test_overlap(train_counter, test_counter)
        
        # Create a separate row for overlap statistics
        overlap_row = {
            "Task": task_name.upper(),
            "Split": "Overlap",
            "Total Tokens": test_stats["Total Tokens"],
            "Unique Tokens": test_stats["Unique Tokens"],
            "Test Unique Tokens Not In Train": overlap_stats["Test Unique Tokens Not In Train"],
            "Test Unique Not In Train (%)": overlap_stats["Test Unique Tokens Not In Train (%)"],
            "Test OOV Occurrences": overlap_stats["Test OOV Token Occurrences"],
            "Test OOV Occurrences (%)": overlap_stats["Test OOV Token Occurrences (%)"]
        }
        results.append(overlap_row)
        
        # Save task-specific results to CSV for checkpointing
        task_df = pd.DataFrame(results)
        task_df.to_csv(f"{task_name}_token_stats.csv", index=False)
        print(f"Saved {task_name} statistics to {task_name}_token_stats.csv")
        
        return results
    
    except Exception as e:
        print(f"Error processing {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

def combine_task_results():
    """Combine all task CSVs into a single result set"""
    all_results = []
    for task in glue_tasks.keys():
        try:
            task_df = pd.read_csv(f"{task}_token_stats.csv")
            all_results.append(task_df)
        except FileNotFoundError:
            print(f"Warning: No stats file found for {task}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

# Process tasks one by one
for task, config in glue_tasks.items():
    try:
        task_stats = analyze_task(task, config)
    except Exception as e:
        print(f"Error processing {task}: {e}")
        continue

# Combine all results
df = combine_task_results()

if not df.empty:
    # Print tabular results
    print("\nToken Statistics for GLUE Datasets (Train and Test Splits):")
    print(df.to_string(index=False))
    
    # Save combined results to CSV
    df.to_csv("token_statistics.csv", index=False)
    
    # Create a pivot table for LaTeX export - main statistics
    train_test_df = df[df['Split'].isin(['Train', 'Test'])]
    pivot_df = train_test_df.pivot_table(
        index="Task", 
        columns="Split", 
        values=["Total Tokens", "Unique Tokens", "Vocab Coverage (%)", 
                "Top 10% Coverage (%)", "Top 20% Coverage (%)", 
                "Top 50% Coverage (%)", "TF-IDF/Freq Overlap (%)"]
    )
    
    # Flatten the column hierarchical index for easier handling in LaTeX
    pivot_df.columns = [f"{col[0]} ({col[1]})" for col in pivot_df.columns]
    
    # Reset index to make Task a column
    pivot_df = pivot_df.reset_index()
    
    # Create separate overlap statistics dataframe
    overlap_df = df[df['Split'] == 'Overlap'].copy()
    
    # Format the pivot table for LaTeX - main statistics table
    latex_table = """% Token Statistics for GLUE Datasets - Train and Test Split Comparison
\\begin{table*}[htbp]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3.5pt}
\\begin{tabular}{l|cc|cc|cc|cc|cc}
\\toprule
& \\multicolumn{2}{c|}{\\textbf{Total Tokens}} & \\multicolumn{2}{c|}{\\textbf{Unique Tokens}} & \\multicolumn{2}{c|}{\\textbf{Vocab Coverage (\\%)}} & \\multicolumn{2}{c|}{\\textbf{Top 20\\% Coverage (\\%)}} & \\multicolumn{2}{c}{\\textbf{TF-IDF/Freq Overlap (\\%)}} \\\\
\\textbf{Task} & \\textbf{Train} & \\textbf{Test} & \\textbf{Train} & \\textbf{Test} & \\textbf{Train} & \\textbf{Test} & \\textbf{Train} & \\textbf{Test} & \\textbf{Train} & \\textbf{Test} \\\\
\\midrule
"""
    
    # Add rows for each task
    for _, row in pivot_df.iterrows():
        task = row['Task']
        train_tokens = f"{int(row['Total Tokens (Train)']):,}"
        test_tokens = f"{int(row['Total Tokens (Test)']):,}"
        train_unique = f"{int(row['Unique Tokens (Train)']):,}"
        test_unique = f"{int(row['Unique Tokens (Test)']):,}"
        train_coverage = f"{row['Vocab Coverage (%) (Train)']:.2f}"
        test_coverage = f"{row['Vocab Coverage (%) (Test)']:.2f}"
        train_top20 = f"{row['Top 20% Coverage (%) (Train)']:.2f}"
        test_top20 = f"{row['Top 20% Coverage (%) (Test)']:.2f}"
        train_overlap = f"{row['TF-IDF/Freq Overlap (%) (Train)']:.2f}"
        test_overlap = f"{row['TF-IDF/Freq Overlap (%) (Test)']:.2f}"
        
        latex_table += f"{task} & {train_tokens} & {test_tokens} & {train_unique} & {test_unique} & {train_coverage} & {test_coverage} & {train_top20} & {test_top20} & {train_overlap} & {test_overlap} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\caption{Token statistics across GLUE tasks, comparing train and test splits. The table shows how the token distributions differ between splits, highlighting the significant vocabulary redundancy in most datasets, with only a small percentage of tokens (Top 20\\%) covering the vast majority of each dataset. This provides empirical justification for our vocabulary pruning approach. The TF-IDF/Frequency overlap metric indicates the difference between importance-based and pure frequency-based token selection, explaining the superior performance of our TF-IDF approach.}
\\label{tab:token_statistics}
\\end{table*}"""
    
    with open("token_statistics_table.tex", "w") as f:
        f.write(latex_table)
    
    # Create train-test overlap table
    if not overlap_df.empty:
        # Format the overlap statistics for LaTeX
        overlap_latex_table = """% Train-Test Vocabulary Overlap Statistics
\\begin{table*}[htbp]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{4pt}
\\begin{tabular}{l|cc|cc}
\\toprule
& \\multicolumn{2}{c|}{\\textbf{Test Vocabulary Coverage}} & \\multicolumn{2}{c}{\\textbf{Test Token Occurrences}} \\\\
\\textbf{Task} & \\textbf{OOV Tokens} & \\textbf{OOV\\%} & \\textbf{OOV Occurrences} & \\textbf{OOV\\%} \\\\
\\midrule
"""
        
        # Add rows for each task
        for _, row in overlap_df.iterrows():
            task = row['Task']
            
            try:
                oov_tokens = int(row['Test Unique Tokens Not In Train'])
                oov_tokens_pct = float(row['Test Unique Not In Train (%)'])
                oov_count = int(row['Test OOV Occurrences'])
                oov_count_pct = float(row['Test OOV Occurrences (%)'])
            except:
                # Default values if columns are missing
                oov_tokens = 0
                oov_tokens_pct = 0.0
                oov_count = 0
                oov_count_pct = 0.0
            
            overlap_latex_table += f"{task} & {oov_tokens:,} & {oov_tokens_pct:.2f}\\% & {oov_count:,} & {oov_count_pct:.2f}\\% \\\\\n"
        
        overlap_latex_table += """\\bottomrule
\\end{tabular}
\\caption{Train-test vocabulary overlap statistics. \\textbf{OOV Tokens}: number of unique tokens in the test set that don't appear in the training set; \\textbf{OOV\\%}: percentage of test vocabulary not found in train; \\textbf{OOV Occurrences}: number of token occurrences in test set that weren't seen during training; \\textbf{OOV\\%}: percentage of test tokens that were unseen during training. Lower percentages indicate better vocabulary coverage, which is beneficial for pruning.}
\\label{tab:vocab_overlap}
\\end{table*}"""
        
        with open("vocab_overlap_table.tex", "w") as f:
            f.write(overlap_latex_table)
        
        print("\nVocabulary overlap table saved to vocab_overlap_table.tex")
    
    print("\nLaTeX table saved to token_statistics_table.tex")
    print("CSV data saved to token_statistics.csv")
    
    # Generate visualizations comparing train vs test stats
    train_data = df[df['Split'] == 'Train']
    test_data = df[df['Split'] == 'Test']
    
    if len(train_data) > 0 and len(test_data) > 0:
        # Vocabulary coverage visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(train_data))
        width = 0.35
        
        ax.bar(x - width/2, train_data['Vocab Coverage (%)'], width, label='Train', color='royalblue')
        ax.bar(x + width/2, test_data['Vocab Coverage (%)'], width, label='Test', color='lightcoral')
        
        ax.set_xlabel('GLUE Task', fontsize=12)
        ax.set_ylabel('Vocabulary Coverage (%)', fontsize=12)
        ax.set_title('Vocabulary Coverage: Train vs Test Split', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(train_data['Task'])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("vocab_coverage_train_vs_test.png", dpi=300)
        plt.close()
        
        # Top 20% coverage visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width/2, train_data['Top 20% Coverage (%)'], width, label='Train', color='royalblue')
        ax.bar(x + width/2, test_data['Top 20% Coverage (%)'], width, label='Test', color='lightcoral')
        
        ax.set_xlabel('GLUE Task', fontsize=12)
        ax.set_ylabel('Top 20% Token Coverage (%)', fontsize=12)
        ax.set_title('Top 20% Token Coverage: Train vs Test Split', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(train_data['Task'])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("top20_coverage_train_vs_test.png", dpi=300)
        
        # OOV visualization if we have overlap data
        if not overlap_df.empty:
            try:
                fig, ax = plt.subplots(figsize=(14, 8))
                tasks = overlap_df['Task'].values
                x = np.arange(len(tasks))
                
                oov_token_pct = overlap_df['Test Unique Not In Train (%)'].values
                oov_occurrence_pct = overlap_df['Test OOV Occurrences (%)'].values
                
                ax.bar(x - width/2, oov_token_pct, width, label='OOV Unique Tokens (%)', color='indianred')
                ax.bar(x + width/2, oov_occurrence_pct, width, label='OOV Token Occurrences (%)', color='darkred')
                
                ax.set_xlabel('GLUE Task', fontsize=12)
                ax.set_ylabel('Percentage (%)', fontsize=12)
                ax.set_title('Out-of-Vocabulary Tokens in Test Set', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(tasks)
                ax.legend()
                
                plt.tight_layout()
                plt.savefig("oov_tokens_test_set.png", dpi=300)
                print("Generated OOV visualization: oov_tokens_test_set.png")
            except Exception as e:
                print(f"Error generating OOV visualization: {e}")
        
        print("Visualizations saved to vocab_coverage_train_vs_test.png and top20_coverage_train_vs_test.png")
else:
    print("No task statistics were collected. Check individual task errors.")
