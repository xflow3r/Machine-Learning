#!/usr/bin/env python3
"""
Generate tables and charts from experiment results.

Usage:
    python tables.py --chart all          # Generate all charts (default)
    python tables.py --chart accuracy     # Chart 1: Accuracy/F1 table with color gradient
    python tables.py --chart bar          # Chart 2: Accuracy bar chart by model+dataset
    python tables.py --chart category     # Chart 3: Category comparison (side-by-side)
    python tables.py --chart overview     # Chart 4: Full results table with highlights
    python tables.py --chart time         # Chart 5: Time breakdown analysis
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_CSV = os.path.join(PROJECT_ROOT, 'results', 'tables', 'results.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')


def load_results():
    """Load results.csv and create a unique model identifier for CNNs with augmentation."""
    df = pd.read_csv(RESULTS_CSV)
    
    # Create unique model name that distinguishes augmented CNNs
    def get_model_label(row):
        if row['model'].startswith('cnn') and row['augmented']:
            return f"{row['model']}_aug"
        return row['model']
    
    df['model_label'] = df.apply(get_model_label, axis=1)
    df['combo'] = df['dataset'] + ' + ' + df['model_label']
    
    return df


def create_color_gradient(values, reverse=False):
    """
    Create color gradient from green (good) to yellow (medium) to red (bad).
    
    Args:
        values: Array of values to map to colors
        reverse: If True, lower values are better (e.g., for time)
    """
    # Normalize values to 0-1
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        normalized = np.full_like(values, 0.5, dtype=float)
    else:
        normalized = (values - vmin) / (vmax - vmin)
    
    if reverse:
        normalized = 1 - normalized
    
    # Create custom colormap: red -> yellow -> green
    colors = ['#ff4444', '#ffff44', '#44ff44']  # red, yellow, green
    cmap = mcolors.LinearSegmentedColormap.from_list('ryg', colors)
    
    return [cmap(v) for v in normalized]


def chart_accuracy_f1_table(df):
    """
    Chart 1: Generate accuracy and F1-score table with color gradient.
    Cells colored green (good) -> yellow (medium) -> red (bad).
    Columns: Model, Fashion-MNIST Accuracy, Fashion-MNIST F1-Macro, CIFAR-10 Accuracy, CIFAR-10 F1-Macro
    """
    print("Generating Chart 1: Accuracy/F1 Table with Color Gradient...")
    
    # Pivot data: rows are models, columns are dataset metrics
    models = df['model_label'].unique()
    
    # Build pivoted data
    table_data = []
    for model in models:
        row_data = {'model': model}
        for dataset in ['fashion_mnist', 'cifar10']:
            subset = df[(df['model_label'] == model) & (df['dataset'] == dataset)]
            if len(subset) > 0:
                row_data[f'{dataset}_acc'] = subset['accuracy'].values[0]
                row_data[f'{dataset}_f1'] = subset['f1_macro'].values[0]
            else:
                row_data[f'{dataset}_acc'] = None
                row_data[f'{dataset}_f1'] = None
        table_data.append(row_data)
    
    pivot_df = pd.DataFrame(table_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, len(pivot_df) * 0.7 + 2))
    ax.axis('off')
    
    # Prepare table data
    headers = ['Model', 'Fashion-MNIST\nAccuracy', 'Fashion-MNIST\nF1-Macro', 
               'CIFAR-10\nAccuracy', 'CIFAR-10\nF1-Macro']
    
    # Collect all metric values for color gradient calculation
    all_acc_values = []
    all_f1_values = []
    for _, row in pivot_df.iterrows():
        for col in ['fashion_mnist_acc', 'cifar10_acc']:
            if row[col] is not None:
                all_acc_values.append(row[col])
        for col in ['fashion_mnist_f1', 'cifar10_f1']:
            if row[col] is not None:
                all_f1_values.append(row[col])
    
    all_acc_values = np.array(all_acc_values)
    all_f1_values = np.array(all_f1_values)
    
    cell_text = []
    cell_colors = []
    
    for _, row in pivot_df.iterrows():
        row_text = [row['model']]
        row_colors = ['white']
        
        # Fashion-MNIST Accuracy
        val = row['fashion_mnist_acc']
        if val is not None:
            row_text.append(f"{val:.4f}")
            row_colors.append(create_color_gradient(all_acc_values)[list(all_acc_values).index(val)])
        else:
            row_text.append('N/A')
            row_colors.append('lightgray')
        
        # Fashion-MNIST F1
        val = row['fashion_mnist_f1']
        if val is not None:
            row_text.append(f"{val:.4f}")
            row_colors.append(create_color_gradient(all_f1_values)[list(all_f1_values).index(val)])
        else:
            row_text.append('N/A')
            row_colors.append('lightgray')
        
        # CIFAR-10 Accuracy
        val = row['cifar10_acc']
        if val is not None:
            row_text.append(f"{val:.4f}")
            row_colors.append(create_color_gradient(all_acc_values)[list(all_acc_values).index(val)])
        else:
            row_text.append('N/A')
            row_colors.append('lightgray')
        
        # CIFAR-10 F1
        val = row['cifar10_f1']
        if val is not None:
            row_text.append(f"{val:.4f}")
            row_colors.append(create_color_gradient(all_f1_values)[list(all_f1_values).index(val)])
        else:
            row_text.append('N/A')
            row_colors.append('lightgray')
        
        cell_text.append(row_text)
        cell_colors.append(row_colors)
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=['lightgray'] * 5,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold')
    
    plt.title('Model Performance: Accuracy and F1-Score\n(Green=Best, Yellow=Medium, Red=Worst)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'chart1_accuracy_f1_table.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def chart_accuracy_bar(df):
    """
    Chart 2: Bar chart showing accuracy for each model+dataset combo.
    Different colors for each dataset.
    """
    print("Generating Chart 2: Accuracy Bar Chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique models and datasets
    models = df['model_label'].unique()
    datasets = df['dataset'].unique()
    
    x = np.arange(len(models))
    width = 0.35
    
    # Color palette for datasets
    colors = {'fashion_mnist': '#3498db', 'cifar10': '#e74c3c'}
    
    for i, dataset in enumerate(datasets):
        subset = df[df['dataset'] == dataset]
        # Ensure order matches models
        accuracies = [subset[subset['model_label'] == m]['accuracy'].values[0] 
                      if len(subset[subset['model_label'] == m]) > 0 else 0 
                      for m in models]
        
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, accuracies, width, label=dataset, color=colors.get(dataset, f'C{i}'))
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.annotate(f'{acc:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy by Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Dataset')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'chart2_accuracy_bar.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def chart_category_comparison(df):
    """
    Chart 3: Side-by-side bar charts comparing model categories.
    Categories: Simple Baseline (histogram), Powerful Traditional (HOG), Deep Learning (CNN)
    """
    print("Generating Chart 3: Category Comparison (Side-by-Side)...")
    
    def categorize_model(model_label):
        if model_label.startswith('hist_'):
            return 'Simple Baseline\n(Color Histogram)'
        elif model_label.startswith('hog_'):
            return 'Powerful Traditional\n(HOG)'
        elif model_label.startswith('cnn'):
            return 'Deep Learning\n(CNN)'
        return 'Other'
    
    df['category'] = df['model_label'].apply(categorize_model)
    
    # Calculate average accuracy per category per dataset
    # For CNNs, include augmented versions in the average
    category_avg = df.groupby(['dataset', 'category'])['accuracy'].mean().reset_index()
    
    datasets = df['dataset'].unique()
    categories = ['Simple Baseline\n(Color Histogram)', 'Powerful Traditional\n(HOG)', 'Deep Learning\n(CNN)']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # red, orange, green
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = category_avg[category_avg['dataset'] == dataset]
        
        # Ensure correct order
        accuracies = []
        for cat in categories:
            val = subset[subset['category'] == cat]['accuracy'].values
            accuracies.append(val[0] if len(val) > 0 else 0)
        
        bars = ax.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.annotate(f'{acc:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Average Accuracy', fontsize=12)
        ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=9)
    
    plt.suptitle('Category Comparison: Average Accuracy by Model Type', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'chart3_category_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def chart_overview_table(df):
    """
    Chart 4: Full results table with best values highlighted.
    Best accuracy/f1 = highest (bold + yellow)
    Best times = lowest (bold + yellow)
    """
    print("Generating Chart 4: Overview Table with Highlights...")
    
    # Select columns (exclude seed and notes)
    cols = ['dataset', 'model_label', 'accuracy', 'f1_macro', 
            'feature_time', 'train_time', 'test_time', 'total_time']
    table_df = df[cols].copy()
    
    # Determine best values for each metric
    best_values = {
        'accuracy': table_df['accuracy'].max(),
        'f1_macro': table_df['f1_macro'].max(),
        'feature_time': table_df[table_df['feature_time'] > 0]['feature_time'].min() if (table_df['feature_time'] > 0).any() else 0,
        'train_time': table_df['train_time'].min(),
        'test_time': table_df['test_time'].min(),
        'total_time': table_df['total_time'].min(),
    }
    
    # For metrics where higher is better
    higher_better = ['accuracy', 'f1_macro']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, len(table_df) * 0.55 + 2))
    ax.axis('off')
    
    headers = ['Dataset', 'Model', 'Accuracy', 'F1-Macro', 
               'Feature Time (s)', 'Train Time (s)', 'Test Time (s)', 'Total Time (s)']
    
    cell_text = []
    cell_colors = []
    
    highlight_color = '#ffff00'  # Yellow
    normal_color = 'white'
    
    for idx, row in table_df.iterrows():
        row_text = []
        row_colors = []
        
        for col in cols:
            val = row[col]
            
            if col in ['dataset', 'model_label']:
                row_text.append(str(val))
                row_colors.append(normal_color)
            else:
                # Format numeric values
                formatted = f"{val:.4f}" if col in ['accuracy', 'f1_macro'] else f"{val:.2f}"
                
                # Check if this is the best value
                is_best = False
                if col in higher_better:
                    is_best = (val == best_values[col])
                elif col == 'feature_time':
                    # Special case: 0 feature time for CNNs isn't "best"
                    is_best = (val > 0 and val == best_values[col])
                else:
                    is_best = (val == best_values[col])
                
                if is_best:
                    formatted = f"**{formatted}**"  # We'll bold via font properties
                    row_colors.append(highlight_color)
                else:
                    row_colors.append(normal_color)
                
                row_text.append(formatted.replace('**', ''))
        
        cell_text.append(row_text)
        cell_colors.append(row_colors)
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=['lightgray'] * len(headers),
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    
    # Bold headers
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold')
    
    # Bold best values
    for row_idx, row in table_df.iterrows():
        for col_idx, col in enumerate(cols):
            if col in ['dataset', 'model_label']:
                continue
            val = row[col]
            is_best = False
            if col in higher_better:
                is_best = (val == best_values[col])
            elif col == 'feature_time':
                is_best = (val > 0 and val == best_values[col])
            else:
                is_best = (val == best_values[col])
            
            if is_best:
                # row_idx in table is offset by 1 due to header
                cell = table[(list(table_df.index).index(row_idx) + 1, col_idx)]
                cell.set_text_props(weight='bold')
    
    plt.title('Complete Results Overview\n(Yellow + Bold = Best in Column)', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_path = os.path.join(OUTPUT_DIR, 'chart4_overview_table.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def chart_time_breakdown(df):
    """
    Chart 5: Stacked bar chart showing time breakdown.
    Shows share of feature extraction, training, and testing time.
    """
    print("Generating Chart 5: Time Breakdown Analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chart 5a: Absolute time breakdown (stacked bar)
    ax1 = axes[0]
    
    combos = df['combo'].values
    feature_times = df['feature_time'].values
    train_times = df['train_time'].values
    test_times = df['test_time'].values
    
    x = np.arange(len(combos))
    width = 0.6
    
    bars1 = ax1.bar(x, feature_times, width, label='Feature Extraction', color='#3498db')
    bars2 = ax1.bar(x, train_times, width, bottom=feature_times, label='Training', color='#e74c3c')
    bars3 = ax1.bar(x, test_times, width, bottom=feature_times + train_times, label='Testing', color='#2ecc71')
    
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Absolute Time Breakdown', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(combos, rotation=45, ha='right', fontsize=8)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # Chart 5b: Percentage breakdown (stacked bar, normalized to 100%)
    ax2 = axes[1]
    
    total_times = df['total_time'].values
    feature_pct = (feature_times / total_times) * 100
    train_pct = (train_times / total_times) * 100
    test_pct = (test_times / total_times) * 100
    
    bars1 = ax2.bar(x, feature_pct, width, label='Feature Extraction', color='#3498db')
    bars2 = ax2.bar(x, train_pct, width, bottom=feature_pct, label='Training', color='#e74c3c')
    bars3 = ax2.bar(x, test_pct, width, bottom=feature_pct + train_pct, label='Testing', color='#2ecc71')
    
    ax2.set_ylabel('Percentage of Total Time (%)', fontsize=12)
    ax2.set_title('Relative Time Breakdown (%)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(combos, rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Time Analysis: Feature Extraction vs Training vs Testing', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'chart5_time_breakdown.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_charts(df):
    """Generate all charts."""
    chart_accuracy_f1_table(df)
    chart_accuracy_bar(df)
    chart_category_comparison(df)
    chart_overview_table(df)
    chart_time_breakdown(df)


def main():
    parser = argparse.ArgumentParser(
        description='Generate tables and charts from experiment results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tables.py --chart all          # Generate all charts (default)
    python tables.py --chart accuracy     # Chart 1: Accuracy/F1 table
    python tables.py --chart bar          # Chart 2: Accuracy bar chart
    python tables.py --chart category     # Chart 3: Category comparison
    python tables.py --chart overview     # Chart 4: Full results overview
    python tables.py --chart time         # Chart 5: Time breakdown
        """
    )
    parser.add_argument('--chart', type=str, default='all',
                        choices=['all', 'accuracy', 'bar', 'category', 'overview', 'time'],
                        help='Which chart to generate (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"Loading results from: {RESULTS_CSV}")
    df = load_results()
    print(f"Loaded {len(df)} result rows")
    print(f"Models: {df['model_label'].unique().tolist()}")
    print(f"Datasets: {df['dataset'].unique().tolist()}")
    print()
    
    # Generate requested chart(s)
    chart_functions = {
        'accuracy': chart_accuracy_f1_table,
        'bar': chart_accuracy_bar,
        'category': chart_category_comparison,
        'overview': chart_overview_table,
        'time': chart_time_breakdown,
    }
    
    if args.chart == 'all':
        generate_all_charts(df)
    else:
        chart_functions[args.chart](df)
    
    print()
    print("=" * 50)
    print("Chart generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == '__main__':
    main()
