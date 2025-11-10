"""
Visualization Creation for Phishing SVM Results
Exercise 1 - Machine Learning
Creates all visualizations for the Phishing dataset SVM experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("CREATING VISUALIZATIONS FOR PHISHING DATASET")
print("="*80)
print(f"Script directory: {SCRIPT_DIR}")

# Load results
results_path = os.path.join(SCRIPT_DIR, 'phishing_results.csv')
print(f"Loading results from: {results_path}")

if not os.path.exists(results_path):
    print(f"\n✗ ERROR: Results file not found!")
    print(f"  Please run 'phishing_svm.py' first to generate results.")
    exit(1)

results_df = pd.read_csv(results_path)

# Remove any rows with errors
results_df = results_df[results_df['accuracy'].notna()]

if len(results_df) == 0:
    print("\n✗ ERROR: No valid results found in CSV!")
    exit(1)

print(f"✓ Loaded {len(results_df)} experiment results")
print(f"\nOutput directory: {SCRIPT_DIR}\n")


# ============================================================================
# 1. ACCURACY COMPARISON: LINEAR VS RBF KERNELS
# ============================================================================
print("1. Creating kernel comparison plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Group by kernel
linear_results = results_df[results_df['kernel'] == 'linear']
rbf_results = results_df[results_df['kernel'] == 'rbf']

# Get best accuracy for each kernel
linear_best = linear_results['accuracy'].max() if len(linear_results) > 0 else 0
rbf_best = rbf_results['accuracy'].max() if len(rbf_results) > 0 else 0

kernels = ['Linear Kernel', 'RBF Kernel']
accuracies = [linear_best, rbf_best]
colors = ['#3498db', '#e74c3c']

bars = ax.bar(kernels, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Phishing Dataset: Linear vs RBF Kernel Performance', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'phishing_viz_1_kernel_comparison.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")


# ============================================================================
# 2. PARAMETER SENSITIVITY: EFFECT OF C
# ============================================================================
print("\n2. Creating parameter sensitivity plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines for each kernel
for kernel, color, marker in [('linear', '#3498db', 'o'), ('rbf', '#e74c3c', 's')]:
    kernel_data = results_df[results_df['kernel'] == kernel].copy()
    kernel_data = kernel_data.sort_values('C')
    
    ax.plot(kernel_data['C'], kernel_data['accuracy'], 
            marker=marker, label=kernel.upper(), linewidth=2.5, 
            markersize=10, color=color)
    
    # Add value labels
    for x, y in zip(kernel_data['C'], kernel_data['accuracy']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('C Parameter', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Phishing Dataset: Parameter Sensitivity (C)', 
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.8, 0.92)

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'phishing_viz_2_parameter_sensitivity.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")


# ============================================================================
# 3. TRAINING TIME COMPARISON
# ============================================================================
print("\n3. Creating training time comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

# Group by kernel and get mean training time
time_by_kernel = results_df.groupby('kernel')['train_time'].agg(['mean', 'std']).reset_index()

colors_dict = {'linear': '#3498db', 'rbf': '#e74c3c'}
bar_colors = [colors_dict[k] for k in time_by_kernel['kernel']]

bars = ax.bar(time_by_kernel['kernel'], time_by_kernel['mean'], 
              yerr=time_by_kernel['std'], capsize=5,
              color=bar_colors, edgecolor='black', linewidth=2, alpha=0.8)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}s',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_xlabel('Kernel Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Phishing Dataset: Training Time by Kernel', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'phishing_viz_3_training_time.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")


# ============================================================================
# 4. HEATMAP: ALL RESULTS
# ============================================================================
print("\n4. Creating results heatmap...")

# Create pivot table
pivot_data = results_df.pivot_table(
    values='accuracy',
    index='kernel',
    columns='C',
    aggfunc='max'
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
            vmin=0.8, vmax=0.95, cbar_kws={'label': 'Accuracy'},
            linewidths=2, linecolor='black', ax=ax, annot_kws={'fontsize': 11})

ax.set_title('Phishing Dataset: SVM Performance Heatmap', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('C Parameter', fontsize=12, fontweight='bold')
ax.set_ylabel('Kernel Type', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'phishing_viz_4_heatmap.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")


# ============================================================================
# 5. COMPREHENSIVE METRICS TABLE
# ============================================================================
print("\n5. Creating comprehensive metrics table...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['Kernel', 'C', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (s)'])

for _, row in results_df.iterrows():
    table_data.append([
        row['kernel'].upper(),
        f"{row['C']:.1f}",
        f"{row['accuracy']:.4f}",
        f"{row['precision']:.4f}" if pd.notna(row['precision']) else 'N/A',
        f"{row['recall']:.4f}" if pd.notna(row['recall']) else 'N/A',
        f"{row['f1']:.4f}" if pd.notna(row['f1']) else 'N/A',
        f"{row['train_time']:.4f}"
    ])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center',
                colColours=['#3498db']*7)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header row
for i in range(7):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors and highlight best accuracy
best_acc_idx = results_df['accuracy'].idxmax()
for i in range(1, len(table_data)):
    for j in range(7):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        # Highlight best result
        if i-1 == best_acc_idx:
            table[(i, j)].set_facecolor('#abebc6')
            table[(i, j)].set_text_props(weight='bold')

ax.set_title('Phishing Dataset: Complete SVM Results (Best Result Highlighted)', 
            fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'phishing_viz_5_metrics_table.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")


# ============================================================================
# 6. SUMMARY DASHBOARD
# ============================================================================
print("\n6. Creating summary dashboard...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 6a. Best accuracy per kernel (top left)
ax1 = fig.add_subplot(gs[0, 0])
kernel_best = results_df.groupby('kernel')['accuracy'].max().reset_index()
colors_list = ['#3498db' if k == 'linear' else '#e74c3c' for k in kernel_best['kernel']]
bars = ax1.bar(kernel_best['kernel'], kernel_best['accuracy'], 
              color=colors_list, edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('Best Accuracy', fontweight='bold')
ax1.set_title('Best Accuracy by Kernel', fontweight='bold', fontsize=12)
ax1.set_ylim(0.8, 0.95)
ax1.grid(True, alpha=0.3, axis='y')

# 6b. Training time by C parameter (top right)
ax2 = fig.add_subplot(gs[0, 1])
for kernel, color in [('linear', '#3498db'), ('rbf', '#e74c3c')]:
    kernel_data = results_df[results_df['kernel'] == kernel]
    ax2.plot(kernel_data['C'], kernel_data['train_time'], 
            marker='o', label=kernel.upper(), linewidth=2, color=color)
ax2.set_xlabel('C Parameter', fontweight='bold')
ax2.set_ylabel('Training Time (s)', fontweight='bold')
ax2.set_title('Training Time vs C', fontweight='bold', fontsize=12)
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 6c. Accuracy by configuration (middle)
ax3 = fig.add_subplot(gs[1, :])
x = range(len(results_df))
colors_bars = ['#3498db' if k == 'linear' else '#e74c3c' for k in results_df['kernel']]
bars = ax3.bar(x, results_df['accuracy'], color=colors_bars, 
              edgecolor='black', linewidth=1.5)
ax3.axhline(y=results_df['accuracy'].mean(), color='green', 
           linestyle='--', label=f'Mean: {results_df["accuracy"].mean():.3f}')
ax3.set_xlabel('Configuration', fontweight='bold')
ax3.set_ylabel('Accuracy', fontweight='bold')
ax3.set_title('Accuracy Across All Configurations', fontweight='bold', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels([f"{r['kernel'][:3].upper()}\nC={r['C']}" 
                     for _, r in results_df.iterrows()], fontsize=8)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 6d. Dataset info (bottom left)
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('tight')
ax4.axis('off')
dataset_info = [
    ['Dataset', 'Phishing'],
    ['Samples', '1,353'],
    ['Features', '9 (categorical)'],
    ['Classes', '3'],
    ['Train/Val Split', '80/20'],
    ['Best Kernel', results_df.loc[results_df['accuracy'].idxmax(), 'kernel'].upper()],
    ['Best C', f"{results_df.loc[results_df['accuracy'].idxmax(), 'C']:.1f}"],
]
table = ax4.table(cellText=dataset_info, cellLoc='left', loc='center',
                 colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
for i in range(len(dataset_info)):
    table[(i, 0)].set_facecolor('#e8f4f8')
    table[(i, 0)].set_text_props(weight='bold')
ax4.set_title('Dataset Information', fontweight='bold', fontsize=12, pad=20)

# 6e. Summary statistics (bottom right)
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('tight')
ax5.axis('off')
summary_stats = [
    ['Metric', 'Value'],
    ['Best Accuracy', f"{results_df['accuracy'].max():.4f}"],
    ['Worst Accuracy', f"{results_df['accuracy'].min():.4f}"],
    ['Mean Accuracy', f"{results_df['accuracy'].mean():.4f}"],
    ['Std Dev', f"{results_df['accuracy'].std():.4f}"],
    ['Fastest Training', f"{results_df['train_time'].min():.4f}s"],
    ['Slowest Training', f"{results_df['train_time'].max():.4f}s"],
]
table2 = ax5.table(cellText=summary_stats[1:], colLabels=summary_stats[0],
                  cellLoc='center', loc='center', colWidths=[0.6, 0.4])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2.5)
table2[(0, 0)].set_facecolor('#3498db')
table2[(0, 1)].set_facecolor('#3498db')
table2[(0, 0)].set_text_props(weight='bold', color='white')
table2[(0, 1)].set_text_props(weight='bold', color='white')
ax5.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)

plt.suptitle('Phishing Dataset: SVM Classification Summary', 
            fontsize=16, fontweight='bold', y=0.98)

output_path = os.path.join(SCRIPT_DIR, 'phishing_viz_6_summary_dashboard.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*80)
print(f"\nOutput directory: {SCRIPT_DIR}")
print("\nGenerated files:")
print("  1. phishing_viz_1_kernel_comparison.png     - Linear vs RBF performance")
print("  2. phishing_viz_2_parameter_sensitivity.png - Effect of C parameter")
print("  3. phishing_viz_3_training_time.png         - Training time by kernel")
print("  4. phishing_viz_4_heatmap.png               - Complete results heatmap")
print("  5. phishing_viz_5_metrics_table.png         - Detailed metrics table")
print("  6. phishing_viz_6_summary_dashboard.png     - Complete summary dashboard")
print("\nAll images saved at 300 DPI for high-quality report inclusion!")
print("="*80)

# Display best result
best_idx = results_df['accuracy'].idxmax()
best_result = results_df.loc[best_idx]
print("\n" + "="*80)
print("BEST MODEL SUMMARY")
print("="*80)
print(f"Kernel: {best_result['kernel'].upper()}")
print(f"C: {best_result['C']}")
print(f"Accuracy: {best_result['accuracy']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"F1-Score: {best_result['f1']:.4f}")
print(f"Training Time: {best_result['train_time']:.4f}s")
print("="*80)