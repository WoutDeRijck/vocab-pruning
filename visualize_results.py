import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

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

# Read CSV data more carefully
datasets = []
paper_results = []
no_pruning_test = []
train_tokens_test = []
random_test = []
clustering_test = []
frequency_test = []

# Also track training performance
no_pruning_train = []
train_tokens_train = []
random_train = []
clustering_train = []
frequency_train = []

with open('Results - Blad1.csv', 'r') as f:
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

# Create DataFrames for plotting train and test results
test_df = pd.DataFrame({
    'PAPER': paper_results,
    'NO PRUNING': no_pruning_test,
    'TRAIN TOKENS ONLY': train_tokens_test,
    'RANDOM': random_test,
    'CLUSTERING': clustering_test,
    'FREQUENCY': frequency_test
}, index=datasets)

train_df = pd.DataFrame({
    'NO PRUNING': no_pruning_train,
    'TRAIN TOKENS ONLY': train_tokens_train,
    'RANDOM': random_train,
    'CLUSTERING': clustering_train,
    'FREQUENCY': frequency_train
}, index=datasets)

# Print the DataFrame to verify data
print("Test Results DataFrame:")
print(test_df)

# Define the list of methods for consistent reference
methods = ['NO PRUNING', 'TRAIN TOKENS ONLY', 'RANDOM', 'CLUSTERING', 'FREQUENCY']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
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
plt.figure(figsize=(14, 8))
x = np.arange(len(test_df))
width = 0.14
multiplier = 0

for method in methods:
    offset = width * multiplier
    plt.bar(x + offset, test_df[method], width, label=method, color=method_colors[method])
    multiplier += 1

# Add paper results as horizontal lines for each dataset
for i, (dataset, row) in enumerate(test_df.iterrows()):
    plt.plot([i - width*2, i + width*5], [row['PAPER'], row['PAPER']], 'k--', alpha=0.7)

# Add labels and title
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Test Performance', fontsize=12)
plt.title('Comparison of Pruning Methods Across GLUE Datasets', fontsize=14)
plt.xticks(x + width * 2, test_df.index, rotation=45, fontsize=10)
plt.legend(loc='lower left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for i, method in enumerate(methods):
    for j, value in enumerate(test_df[method].values):
        if value > 0:  # Only add label if value is greater than 0
            plt.text(j + width * (i - 0.5), value + 0.01, 
                     f'{value:.3f}', ha='center', va='bottom', 
                     fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig('plots/pruning_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 2: Line chart for easy comparison
plt.figure(figsize=(14, 8))

# Create x positions for each dataset
x_positions = np.arange(len(test_df))

# Plot lines for each method
for method in methods:
    values = test_df[method]
    plt.plot(x_positions, values, 'o-', label=method, linewidth=2, markersize=8, color=method_colors[method])

# Add paper results
plt.plot(x_positions, test_df['PAPER'], 'k--', label='Paper Results', linewidth=2, markersize=10)

plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Test Performance', fontsize=12)
plt.title('Performance Comparison of Different Pruning Methods', fontsize=14)
plt.xticks(x_positions, test_df.index, rotation=45, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('plots/pruning_comparison_line.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 3: Bar chart showing performance drop from baseline
plt.figure(figsize=(14, 8))
drop_df = drop_df.loc[test_df.index]  # Sort in the same order

x = np.arange(len(drop_df))
width = 0.18
multiplier = 0

for method in methods[1:]:  # Skip NO PRUNING
    offset = width * multiplier
    plt.bar(x + offset, drop_df[method], width, label=method, color=method_colors[method])
    multiplier += 1

plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Performance Drop (%)', fontsize=12)
plt.title('Performance Drop Compared to No Pruning', fontsize=14)
plt.xticks(x + width * 1.5, drop_df.index, rotation=45, fontsize=10)
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

# VISUALIZATION 4: Train vs Test performance comparison (for generalization assessment)
for method in methods:
    plt.figure(figsize=(12, 6))
    
    # Set up the bar positions
    x = np.arange(len(test_df))
    width = 0.35
    
    # Create the bars
    plt.bar(x - width/2, train_df[method], width, label='Train', color='#1f77b4', alpha=0.7)
    plt.bar(x + width/2, test_df[method], width, label='Test', color='#ff7f0e', alpha=0.7)
    
    # Add paper results as points
    plt.scatter(x, test_df['PAPER'], marker='*', s=100, color='black', label='Paper', zorder=3)
    
    # Add labels and title
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    plt.title(f'Train vs Test Performance: {method}', fontsize=14)
    plt.xticks(x, test_df.index, rotation=45, fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Calculate train-test differences
    diff = train_df[method] - test_df[method]
    
    # Add value labels and train-test difference
    for i in range(len(test_df)):
        plt.text(i - width/2, train_df[method].iloc[i] + 0.01, 
                 f'{train_df[method].iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, test_df[method].iloc[i] + 0.01, 
                 f'{test_df[method].iloc[i]:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add difference indicator
        if abs(diff.iloc[i]) > 0.01:
            plt.text(i, min(train_df[method].iloc[i], test_df[method].iloc[i]) - 0.04,
                     f'Δ: {diff.iloc[i]:.3f}', ha='center', color='red' if diff.iloc[i] > 0 else 'green')
    
    plt.tight_layout()
    plt.savefig(f'plots/train_test_{method.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# VISUALIZATION 5: Performance relative to published results
plt.figure(figsize=(14, 8))
paper_diff_df = paper_diff_df.loc[test_df.index]  # Sort in the same order

x = np.arange(len(paper_diff_df))
width = 0.14
multiplier = 0

for method in methods:
    offset = width * multiplier
    plt.bar(x + offset, paper_diff_df[method], width, label=method, color=method_colors[method])
    multiplier += 1

plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Performance Relative to Paper (%)', fontsize=12)
plt.title('Performance Compared to Published Results', fontsize=14)
plt.xticks(x + width * 2, paper_diff_df.index, rotation=45, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Add value labels above/below bars
for i, method in enumerate(methods):
    for j, value in enumerate(paper_diff_df[method].values):
        if abs(value) > 1:  # Only add label if difference is more than 1%
            plt.text(j + width * (i - 0.5), value + (0.5 if value >= 0 else -1.5), 
                     f'{value:.1f}', ha='center', va='bottom', 
                     fontsize=8)

plt.tight_layout()
plt.savefig('plots/paper_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 6: Summary table for average performance
# Calculate average performance across all datasets
avg_performance = pd.DataFrame({
    'Method': methods + ['PAPER'],
    'Avg Test Performance': [test_df[method].mean() for method in methods] + [test_df['PAPER'].mean()],
    'Avg Drop from Baseline (%)': [((test_df['NO PRUNING'] - test_df[method]) * 100).mean() if method != 'NO PRUNING' else 0 for method in methods] + [((test_df['NO PRUNING'] - test_df['PAPER']) * 100).mean()],
    'Avg Gap from Paper (%)': [((test_df[method] - test_df['PAPER']) * 100).mean() for method in methods] + [0]
})

# Sort by average performance
avg_performance = avg_performance.sort_values('Avg Test Performance', ascending=False)

print("\nAverage Performance Summary:")
print(avg_performance)

# VISUALIZATION 7: Average performance bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(avg_performance['Method'], avg_performance['Avg Test Performance'], 
              color=[method_colors.get(m, 'gray') for m in avg_performance['Method']])

plt.xlabel('Method', fontsize=12)
plt.ylabel('Average Performance', fontsize=12)
plt.title('Average Performance Across All Datasets', fontsize=14)
plt.ylim(0.6, 0.9)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('plots/average_performance.png', dpi=300, bbox_inches='tight')
plt.show() 