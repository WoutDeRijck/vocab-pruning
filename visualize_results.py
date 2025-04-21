import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from utils import glue_tasks  # Import the glue_tasks dictionary that contains dataset sizes

# Function to properly parse a value like "0,65" to 0.65
def parse_value(val):
    if not val or val == '""':
        return 0.0
    # Remove quotes and replace comma with dot
    val = val.replace('"', '').replace(',', '.')
    try:
        return float(val)
    except ValueError:
        return 0.0

# Ensure output directory exists
os.makedirs('plots', exist_ok=True)

# Create a mapping of datasets to their sizes from the glue_tasks dictionary
dataset_sizes = {}
for task_name, task_data in glue_tasks.items():
    if "-" in task_name:  # Handle mnli-matched and mnli-mismatched
        dataset_name = task_name.split("-")[0].upper()
    else:
        dataset_name = task_name.upper()
    dataset_sizes[dataset_name] = task_data["size"]

# Read CSV data more carefully
datasets = []
paper_results = []
no_pruning_test = []
train_tokens_test = []
random_test = []
clustering_test = []
frequency_test = []
importance_test = []
frequency_oov_test = []
importance_oov_test = []

# Also track training performance
no_pruning_train = []
train_tokens_train = []
random_train = []
clustering_train = []
frequency_train = []
importance_train = []

with open('results/SUMMARY.csv', 'r') as f:
    reader = csv.reader(f)
    # Skip header rows
    next(reader)  # Skip first header row
    next(reader)  # Skip second header row
    
    for row in reader:
        # Check if this is a valid data row
        if len(row) > 0 and row[0]:
            datasets.append(row[0])
            paper_results.append(parse_value(row[1]))
            
            # Training results
            no_pruning_train.append(parse_value(row[2]))
            train_tokens_train.append(parse_value(row[4]))
            random_train.append(parse_value(row[6]))
            if len(row) > 8:
                clustering_train.append(parse_value(row[8]))
            else:
                clustering_train.append(0.0)
            if len(row) > 10:
                frequency_train.append(parse_value(row[10]))
            else:
                frequency_train.append(0.0)
            if len(row) > 12:
                importance_train.append(parse_value(row[12]))
            else:
                importance_train.append(0.0)
            
            # Test results
            no_pruning_test.append(parse_value(row[3]))
            train_tokens_test.append(parse_value(row[5]))
            random_test.append(parse_value(row[7]))
            if len(row) > 9:
                clustering_test.append(parse_value(row[9]))
            else:
                clustering_test.append(0.0)
            if len(row) > 11:
                frequency_test.append(parse_value(row[11]))
            else:
                frequency_test.append(0.0)
            if len(row) > 13:
                importance_test.append(parse_value(row[13]))
            else:
                importance_test.append(0.0)
            if len(row) > 14:
                frequency_oov_test.append(parse_value(row[14]))
            else:
                frequency_oov_test.append(0.0)
            if len(row) > 15:
                importance_oov_test.append(parse_value(row[15]))
            else:
                importance_oov_test.append(0.0)

# Create DataFrames for plotting train and test results
test_df = pd.DataFrame({
    'PAPER': paper_results,
    'NO PRUNING': no_pruning_test,
    'TRAIN TOKENS ONLY': train_tokens_test,
    'RANDOM': random_test,
    'CLUSTERING': clustering_test,
    'FREQUENCY': frequency_test,
    'IMPORTANCE': importance_test,
    'FREQUENCY_OOV': frequency_oov_test,
    'IMPORTANCE_OOV': importance_oov_test,
    'SIZE': [dataset_sizes.get(dataset, 'N/A') for dataset in datasets]  # Add dataset sizes
}, index=datasets)

train_df = pd.DataFrame({
    'NO PRUNING': no_pruning_train,
    'TRAIN TOKENS ONLY': train_tokens_train,
    'RANDOM': random_train,
    'CLUSTERING': clustering_train,
    'FREQUENCY': frequency_train,
    'IMPORTANCE': importance_train,
    'SIZE': [dataset_sizes.get(dataset, 'N/A') for dataset in datasets]  # Add dataset sizes
}, index=datasets)

# Print the DataFrame to verify data
print("Test Results DataFrame with dataset sizes:")
print(test_df)

# Define the list of methods for consistent reference
methods = ['NO PRUNING', 'TRAIN TOKENS ONLY', 'RANDOM', 'CLUSTERING', 'FREQUENCY', 'IMPORTANCE', 'FREQUENCY_OOV', 'IMPORTANCE_OOV']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
method_colors = dict(zip(methods, colors))

# Calculate and print performance drop from baseline
print("\nPerformance Drop from No Pruning (in percentage points):")
drop_df = pd.DataFrame()
for method in methods[1:]:  # Skip NO PRUNING
    drop_df[method] = (test_df['NO PRUNING'] - test_df[method]) * 100

print(drop_df)

# Print performance vs paper results
print("\nPerformance Relative to Paper Results (in percentage points):")
paper_diff_df = pd.DataFrame()
for method in methods:
    paper_diff_df[method] = (test_df[method] - test_df['PAPER']) * 100

print(paper_diff_df)

# Sort datasets by baseline performance
test_df = test_df.sort_values(by='NO PRUNING', ascending=False)
train_df = train_df.loc[test_df.index]  # Keep the same ordering

# VISUALIZATION 1: Bar chart comparing test performance across methods
plt.figure(figsize=(18, 10))
x = np.arange(len(test_df))
width = 0.09
multiplier = 0

for method in methods:
    offset = width * multiplier
    plt.bar(x + offset, test_df[method], width, label=method, color=method_colors[method])
    multiplier += 1

# Add paper results as horizontal lines for each dataset
for i, (dataset, row) in enumerate(test_df.iterrows()):
    plt.plot([i - width*3, i + width*8], [row['PAPER'], row['PAPER']], 'k--', alpha=0.7)

# Add labels and title
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Test Performance', fontsize=12)
plt.title('Comparison of Pruning Methods Across GLUE Datasets', fontsize=14)
# Update x-ticks to include dataset sizes
plt.xticks(x + width * 3.5, [f"{dataset}\n({test_df.loc[dataset, 'SIZE']})" for dataset in test_df.index], rotation=45, fontsize=10)
plt.legend(loc='lower left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for i, method in enumerate(methods):
    for j, value in enumerate(test_df[method].values):
        if value > 0:  # Only add label if value is greater than 0
            plt.text(j + width * (i - 0.5), value + 0.01, 
                     f'{value:.3f}', ha='center', va='bottom', 
                     fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig('plots/pruning_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 3: Bar chart showing performance drop from baseline
plt.figure(figsize=(18, 10))
drop_df = drop_df.loc[test_df.index]  # Sort in the same order

x = np.arange(len(drop_df))
width = 0.1
multiplier = 0

for method in methods[1:]:  # Skip NO PRUNING
    offset = width * multiplier
    plt.bar(x + offset, drop_df[method], width, label=method, color=method_colors[method])
    multiplier += 1

plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Performance Drop (%)', fontsize=12)
plt.title('Performance Drop Compared to No Pruning', fontsize=14)
# Update x-ticks to include dataset sizes
plt.xticks(x + width * 3, [f"{dataset}\n({test_df.loc[dataset, 'SIZE']})" for dataset in drop_df.index], rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above/below bars
for i, method in enumerate(methods[1:]):
    for j, value in enumerate(drop_df[method].values):
        if abs(value) > 0.1:  # Only add label if value is significant
            plt.text(j + width * (i - 0.5), value + (0.5 if value >= 0 else -1.5), 
                     f'{value:.1f}', ha='center', va='bottom', 
                     fontsize=8)

plt.tight_layout()
plt.savefig('plots/performance_drop.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 5: Performance relative to published results
plt.figure(figsize=(18, 10))
paper_diff_df = paper_diff_df.loc[test_df.index]  # Sort in the same order

x = np.arange(len(paper_diff_df))
width = 0.09
multiplier = 0

for method in methods:
    offset = width * multiplier
    plt.bar(x + offset, paper_diff_df[method], width, label=method, color=method_colors[method])
    multiplier += 1

plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Performance Relative to Paper (%)', fontsize=12)
plt.title('Performance Compared to Published Results', fontsize=14)
# Update x-ticks to include dataset sizes
plt.xticks(x + width * 3.5, [f"{dataset}\n({test_df.loc[dataset, 'SIZE']})" for dataset in paper_diff_df.index], rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Add value labels above/below bars
for i, method in enumerate(methods):
    for j, value in enumerate(paper_diff_df[method].values):
        if abs(value) > 1:  # Only add label if difference is more than 1%
            plt.text(j + width * (i - 0.5), value + (0.5 if value >= 0 else -1.5), 
                     f'{value:.1f}', ha='center', va='bottom', 
                     fontsize=7)

plt.tight_layout()
plt.savefig('plots/paper_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate average performance across all datasets
# Create a weighted average based on dataset size
size_values = {}
for dataset in test_df.index:
    size_str = dataset_sizes.get(dataset, '0')
    if 'k' in size_str.lower():
        size_values[dataset] = float(size_str.lower().replace('k', '')) * 1000
    else:
        try:
            size_values[dataset] = float(size_str)
        except:
            size_values[dataset] = 0

total_size = sum(size_values.values())

# Create a summary CSV file with dataset sizes
summary_with_sizes = test_df.copy()
# Reformat the output to focus on key metrics and include sizes
summary_output = pd.DataFrame()
summary_output['Dataset'] = test_df.index
summary_output['Size'] = [test_df.loc[dataset, 'SIZE'] for dataset in test_df.index]
for method in methods:
    summary_output[f'{method} Performance'] = [test_df.loc[dataset, method] for dataset in test_df.index]
    if method != 'NO PRUNING':
        summary_output[f'{method} Drop (%)'] = [(test_df.loc[dataset, 'NO PRUNING'] - test_df.loc[dataset, method]) * 100 for dataset in test_df.index]

# Save the summary to a CSV file
summary_output.to_csv('results/SUMMARY_WITH_SIZES.csv', index=False)
print("Generated summary CSV with dataset sizes at: results/SUMMARY_WITH_SIZES.csv")

# VISUALIZATION 8: OOV vs. regular performance comparison
plt.figure(figsize=(14, 8))

# Set up the bar positions
x = np.arange(len(test_df))
width = 0.2

# Create the bars for FREQUENCY and FREQUENCY_OOV
plt.bar(x - width/2, test_df['FREQUENCY'], width, label='FREQUENCY', color=method_colors['FREQUENCY'], alpha=0.7)
plt.bar(x + width/2, test_df['FREQUENCY_OOV'], width, label='FREQUENCY_OOV', color=method_colors['FREQUENCY_OOV'], alpha=0.7)

# Add labels and title
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.title('Regular vs OOV Frequency Pruning Performance', fontsize=14)
# Update x-ticks to include dataset sizes
plt.xticks(x, [f"{dataset}\n({test_df.loc[dataset, 'SIZE']})" for dataset in test_df.index], rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels
for i in range(len(test_df)):
    plt.text(i - width/2, test_df['FREQUENCY'].iloc[i] + 0.01, 
             f'{test_df["FREQUENCY"].iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)
    plt.text(i + width/2, test_df['FREQUENCY_OOV'].iloc[i] + 0.01, 
             f'{test_df["FREQUENCY_OOV"].iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('plots/frequency_vs_oov.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 9: IMPORTANCE vs IMPORTANCE_OOV comparison
plt.figure(figsize=(14, 8))

# Set up the bar positions
x = np.arange(len(test_df))
width = 0.2

# Create the bars for IMPORTANCE and IMPORTANCE_OOV
plt.bar(x - width/2, test_df['IMPORTANCE'], width, label='IMPORTANCE', color=method_colors['IMPORTANCE'], alpha=0.7)
plt.bar(x + width/2, test_df['IMPORTANCE_OOV'], width, label='IMPORTANCE_OOV', color=method_colors['IMPORTANCE_OOV'], alpha=0.7)

# Add labels and title
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.title('Regular vs OOV Importance Pruning Performance', fontsize=14)
# Update x-ticks to include dataset sizes
plt.xticks(x, [f"{dataset}\n({test_df.loc[dataset, 'SIZE']})" for dataset in test_df.index], rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels
for i in range(len(test_df)):
    plt.text(i - width/2, test_df['IMPORTANCE'].iloc[i] + 0.01, 
             f'{test_df["IMPORTANCE"].iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)
    plt.text(i + width/2, test_df['IMPORTANCE_OOV'].iloc[i] + 0.01, 
             f'{test_df["IMPORTANCE_OOV"].iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('plots/importance_vs_oov.png', dpi=300, bbox_inches='tight')
plt.show()

# NEW VISUALIZATIONS USING PARAMETER REDUCTION DATA
# Read the parameter reduction summary data
param_reduction_df = pd.read_csv('results/PARAM_REDUCTION_SUMMARY.csv')

# Extract datasets as index
datasets_param = param_reduction_df.iloc[:, 0].values
param_reduction_df = param_reduction_df.iloc[:, 1:]

# Convert all values to float to avoid string comparison issues
for col in param_reduction_df.columns:
    param_reduction_df[col] = pd.to_numeric(param_reduction_df[col], errors='coerce')

# Convert column headers to MultiIndex
method_groups = ['NO PRUNING', 'TRAIN TOKENS ONLY', 'RANDOM', 'CLUSTERING', 'FREQUENCY', 'IMPORTANCE', 'FREQUENCY_OOV', 'IMPORTANCE_OOV']
subtypes = ['Total', 'Embedding', 'Vocab']
tuples = []

for i, col in enumerate(param_reduction_df.columns):
    method_idx = i // 3
    subtype_idx = i % 3
    if method_idx < len(method_groups):
        tuples.append((method_groups[method_idx], subtypes[subtype_idx]))

param_reduction_df.columns = pd.MultiIndex.from_tuples(tuples)
param_reduction_df.index = datasets_param

# VISUALIZATION 10: Total parameter reduction across methods
plt.figure(figsize=(15, 8))
x = np.arange(len(datasets_param))
width = 0.12
multiplier = 0

# Get methods from the first level of column MultiIndex
pruning_methods = method_groups

for method in pruning_methods:
    if method == 'NO PRUNING':
        continue  # Skip NO PRUNING as it's just zeros
    
    offset = width * multiplier
    # Get 'Total' column for this method 
    try:
        # Access each value individually to avoid array issues
        total_values = [float(param_reduction_df.loc[dataset, (method, 'Total')]) for dataset in datasets_param]
        plt.bar(x + offset, total_values, width, 
                label=method, color=method_colors.get(method, 'gray'))
        multiplier += 1
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not process data for method {method}: {e}")
        continue

plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Total Parameter Reduction (%)', fontsize=12)
plt.title('Total Parameter Reduction Across Pruning Methods', fontsize=14)
# Update x-ticks to include dataset sizes
plt.xticks(x + width * 3, [f"{str(dataset)}\n({dataset_sizes.get(str(dataset).upper(), 'N/A')})" for dataset in datasets_param], rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
i = 0  # Keep track of visible methods
for method in pruning_methods:
    if method == 'NO PRUNING':
        continue  # Skip NO PRUNING
    
    try:
        for j, dataset in enumerate(datasets_param):
            value = float(param_reduction_df.loc[dataset, (method, 'Total')])
            if value > 0:  # Only add label if value is significant
                plt.text(j + width * (i - 0.5), value + 0.5, 
                         f'{value:.1f}', ha='center', va='bottom', 
                         fontsize=8, rotation=90)
        i += 1
    except (KeyError, ValueError, TypeError) as e:
        print(f"Warning: Could not add labels for method {method}: {e}")
        continue

plt.tight_layout()
plt.savefig('plots/total_parameter_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 11: Embedding parameter reduction across methods
plt.figure(figsize=(15, 8))
x = np.arange(len(datasets_param))
width = 0.12
multiplier = 0

for method in pruning_methods:
    if method == 'NO PRUNING':
        continue  # Skip NO PRUNING as it's just zeros
    
    offset = width * multiplier
    try:
        # Access each value individually to avoid array issues
        embedding_values = [float(param_reduction_df.loc[dataset, (method, 'Embedding')]) for dataset in datasets_param]
        plt.bar(x + offset, embedding_values, width, 
                label=method, color=method_colors.get(method, 'gray'))
        multiplier += 1
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not find embedding data for method {method}: {e}")
        continue

plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Embedding Parameter Reduction (%)', fontsize=12)
plt.title('Embedding Parameter Reduction Across Pruning Methods', fontsize=14)
# Update x-ticks to include dataset sizes
plt.xticks(x + width * 3, [f"{str(dataset)}\n({dataset_sizes.get(str(dataset).upper(), 'N/A')})" for dataset in datasets_param], rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
i = 0  # Keep track of visible methods
for method in pruning_methods:
    if method == 'NO PRUNING':
        continue  # Skip NO PRUNING
    
    try:
        for j, dataset in enumerate(datasets_param):
            value = float(param_reduction_df.loc[dataset, (method, 'Embedding')])
            if value > 0:  # Only add label if value is significant
                plt.text(j + width * (i - 0.5), value + 0.5, 
                         f'{value:.1f}', ha='center', va='bottom', 
                         fontsize=8, rotation=90)
        i += 1
    except (KeyError, ValueError, TypeError) as e:
        print(f"Warning: Could not add embedding labels for method {method}: {e}")
        continue

plt.tight_layout()
plt.savefig('plots/embedding_parameter_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

# Enhanced VISUALIZATION: Dataset Size vs Performance with ALL pruning methods
plt.figure(figsize=(14, 8))

# Convert dataset sizes to numeric for plotting
# First, extract just the numeric part and convert to kilobytes
def extract_size_in_kb(size_str):
    if 'k' in size_str.lower():
        return float(size_str.lower().replace('k', ''))
    else:
        try:
            return float(size_str) / 1000  # Convert to kb scale
        except:
            return 0  # Default fallback

# Get size values in numeric form
numeric_sizes = []
for dataset in test_df.index:
    size_str = dataset_sizes.get(dataset, '0')
    numeric_sizes.append(extract_size_in_kb(size_str))

# Use ALL methods instead of just a limited set
plot_methods = methods  # Use all methods

# Create scatter plots for each method
for method in plot_methods:
    plt.scatter(numeric_sizes, test_df[method], s=100, label=method, 
               color=method_colors.get(method, 'gray'), alpha=0.7)
    
    # Add trend line
    z = np.polyfit(np.log10(numeric_sizes), test_df[method], 1)
    p = np.poly1d(z)
    
    # Create a range of x values for the trend line
    x_range = np.logspace(np.log10(min(numeric_sizes)), np.log10(max(numeric_sizes)), 100)
    plt.plot(x_range, p(np.log10(x_range)), "--", color=method_colors.get(method, 'gray'), alpha=0.7)
    
    # Add dataset labels
    for i, dataset in enumerate(test_df.index):
        plt.annotate(dataset, (numeric_sizes[i], test_df[method].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xscale('log')  # Use log scale for dataset size
plt.xlabel('Dataset Size (thousands of examples, log scale)', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.title('Relationship Between Dataset Size and Performance', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Add correlation coefficient as text - moved to bottom right
for i, method in enumerate(plot_methods):
    correlation = np.corrcoef(np.log10(numeric_sizes), test_df[method])[0,1]
    # Position text at bottom right, with each method stacked upward
    plt.text(0.70, 0.05 + 0.04 * i,
             f'{method} correlation with log(size): {correlation:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             color=method_colors.get(method, 'gray'),
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

plt.tight_layout()
plt.savefig('plots/dataset_size_vs_performance_enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 14: Performance vs Parameter Reduction scatterplots for each method
for method in pruning_methods[1:]:  # Skip NO PRUNING
    if method not in test_df.columns:
        continue
        
    plt.figure(figsize=(10, 6))
    
    try:
        # Access each value individually to avoid array issues
        param_reductions = [float(param_reduction_df.loc[dataset, (method, 'Total')]) for dataset in test_df.index]
        perf_drops = ((test_df['NO PRUNING'] - test_df[method]) * 100)
        
        # Create scatter plot with point size representing dataset size (log scale)
        for i, dataset in enumerate(test_df.index):
            # Include dataset size in the annotation
            size_str = dataset_sizes.get(dataset, 'N/A')
            numeric_size = extract_size_in_kb(size_str)
            # Use log of size to determine marker size (with minimum size)
            marker_size = max(50, 20 * np.log10(numeric_size + 10))
            
            plt.scatter(param_reductions[i], perf_drops.iloc[i], 
                       s=marker_size, 
                       color=method_colors.get(method, 'blue'), 
                       alpha=0.7)
            plt.text(param_reductions[i]+0.2, perf_drops.iloc[i]+0.1, 
                    f"{dataset}\n({size_str})", fontsize=9)
        
        plt.xlabel('Parameter Reduction (%)', fontsize=12)
        plt.ylabel('Performance Drop (%)', fontsize=12)
        plt.title(f'Performance Drop vs Parameter Reduction: {method}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(param_reductions, perf_drops, 1)
        p = np.poly1d(z)
        plt.plot(sorted(param_reductions), p(sorted(param_reductions)), "r--", alpha=0.7)
        
        # Add annotation about correlation
        correlation = np.corrcoef(param_reductions, perf_drops)[0,1]
        plt.annotate(f"Correlation: {correlation:.3f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'plots/perf_vs_param_{method.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error plotting scatter for {method}: {e}")
        continue 