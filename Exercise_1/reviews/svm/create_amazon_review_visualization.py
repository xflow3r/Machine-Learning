import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


results_path = os.path.join(SCRIPT_DIR, 'amazon_review_results.csv')
print(f"Loading results from: {results_path}")

if not os.path.exists(results_path):
    print(f"\n✗ ERROR: Results file not found!")
    print(f"  Please run 'amazon_review_svm.py' first to generate results.")
    exit(1)

results_df = pd.read_csv(results_path)

results_df = results_df[results_df['accuracy'].notna()]

if len(results_df) == 0:
    print("\n✗ ERROR: No valid results found in CSV!")
    exit(1)

print(f"✓ Loaded {len(results_df)} experiment results")
print(f"\nOutput directory: {SCRIPT_DIR}\n")

print("1. Creating kernel comparison plot (highlighting dimensionality curse)...")

fig, ax = plt.subplots(figsize=(10, 6))

linear_results = results_df[results_df['kernel'] == 'linear']
rbf_results = results_df[results_df['kernel'] == 'rbf']

linear_best = linear_results['accuracy'].max() if len(linear_results) > 0 else 0
rbf_best = rbf_results['accuracy'].max() if len(rbf_results) > 0 else 0

kernels = ['Linear Kernel\n(Best for High-Dim)', 'RBF Kernel\n(Poor for High-Dim)']
accuracies = [linear_best, rbf_best]
colors = ['#2ecc71', '#e74c3c']

bars = ax.bar(kernels, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}\n({height*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.annotate('Curse of Dimensionality!\nRBF fails in 10,000-dim space', 
            xy=(1, rbf_best), xytext=(1, 0.35),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Amazon Review: Linear vs RBF Performance\n(10,000 Features, 50 Classes)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'amazon_viz_1_kernel_comparison.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

for kernel, color, marker, label in [
    ('linear', '#2ecc71', 'o', 'Linear (High-Dim Winner)'), 
    ('rbf', '#e74c3c', 's', 'RBF (High-Dim Loser)')
]:
    kernel_data = results_df[results_df['kernel'] == kernel].copy()
    kernel_data = kernel_data.sort_values('C')
    
    ax.plot(kernel_data['C'], kernel_data['accuracy'], 
            marker=marker, label=label, linewidth=2.5, 
            markersize=10, color=color)
    
    for x, y in zip(kernel_data['C'], kernel_data['accuracy']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('C Parameter', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Amazon Review: Parameter Sensitivity\n(Note: Linear kernel insensitive to C)', 
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)


ax.text(0.5, 0.95, 
        'Insight: Linear accuracy stable across C values\nRBF improves with C but still poor',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'amazon_viz_2_parameter_sensitivity.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

time_by_kernel = results_df.groupby('kernel')['train_time'].agg(['mean', 'std']).reset_index()

colors_dict = {'linear': '#2ecc71', 'rbf': '#e74c3c'}
bar_colors = [colors_dict[k] for k in time_by_kernel['kernel']]

bars = ax.bar(time_by_kernel['kernel'], time_by_kernel['mean'], 
              yerr=time_by_kernel['std'], capsize=5,
              color=bar_colors, edgecolor='black', linewidth=2, alpha=0.8)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}s',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_xlabel('Kernel Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Amazon Review: Training Time by Kernel\n(600 samples, 10,000 features)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax.text(0.5, 0.95, 
        'Note: Similar training times despite different performance',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'amazon_viz_3_training_time.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()

pivot_data = results_df.pivot_table(
    values='accuracy',
    index='kernel',
    columns='C',
    aggfunc='max'
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
            vmin=0, vmax=0.65, cbar_kws={'label': 'Accuracy'},
            linewidths=2, linecolor='black', ax=ax, annot_kws={'fontsize': 11})

ax.set_title('Amazon Review: SVM Performance Heatmap\n(Green = Better, Red = Worse)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('C Parameter', fontsize=12, fontweight='bold')
ax.set_ylabel('Kernel Type', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'amazon_viz_4_heatmap.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

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
        f"{row['train_time']:.3f}"
    ])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc='center', loc='center',
                colColours=['#3498db']*7)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

for i in range(7):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

best_acc_idx = results_df['accuracy'].idxmax()
for i in range(1, len(table_data)):
    for j in range(7):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

        if i-1 == best_acc_idx:
            table[(i, j)].set_facecolor('#abebc6')
            table[(i, j)].set_text_props(weight='bold')

        elif results_df.iloc[i-1]['accuracy'] < 0.1:
            table[(i, j)].set_facecolor('#f5b7b1')

ax.set_title('Amazon Review: Complete SVM Results\n(Best=Green, Worst=Red)', 
            fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'amazon_viz_5_metrics_table.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

linear_acc = results_df[results_df['kernel'] == 'linear']['accuracy'].values
rbf_acc = results_df[results_df['kernel'] == 'rbf']['accuracy'].values

bp1 = ax1.boxplot([linear_acc, rbf_acc], labels=['Linear', 'RBF'],
                   patch_artist=True, showmeans=True)
bp1['boxes'][0].set_facecolor('#2ecc71')
bp1['boxes'][1].set_facecolor('#e74c3c')

ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Distribution by Kernel', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 0.7)

ax1.text(1, linear_acc.mean(), f'μ={linear_acc.mean():.3f}', 
         ha='right', va='center', fontweight='bold', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(2, rbf_acc.mean(), f'μ={rbf_acc.mean():.3f}', 
         ha='left', va='center', fontweight='bold', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.axis('off')
explanation_text = """
:
Linear SVM >> RBF SVM
"""

ax2.text(0.05, 0.95, explanation_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Amazon Review: Understanding High-Dimensional SVM Performance', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'amazon_viz_6_dimensionality_insight.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()


fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
kernel_best = results_df.groupby('kernel')['accuracy'].max().reset_index()
colors_list = ['#2ecc71' if k == 'linear' else '#e74c3c' for k in kernel_best['kernel']]
bars = ax1.bar(kernel_best['kernel'], kernel_best['accuracy'], 
              color=colors_list, edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax1.set_ylabel('Best Accuracy', fontweight='bold')
ax1.set_title('Best Accuracy by Kernel', fontweight='bold', fontsize=11)
ax1.set_ylim(0, 0.7)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = fig.add_subplot(gs[0, 1])
for kernel, color in [('linear', '#2ecc71'), ('rbf', '#e74c3c')]:
    kernel_data = results_df[results_df['kernel'] == kernel]
    ax2.plot(kernel_data['C'], kernel_data['train_time'], 
            marker='o', label=kernel.upper(), linewidth=2, color=color)
ax2.set_xlabel('C Parameter', fontweight='bold', fontsize=10)
ax2.set_ylabel('Training Time (s)', fontweight='bold', fontsize=10)
ax2.set_title('Training Time vs C', fontweight='bold', fontsize=11)
ax2.set_xscale('log')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
challenges = [
    "Dataset Challenges:",
    "",
    "🔴 Very High-Dimensional",
    "   10,000 features",
    "",
    "🔴 Many Classes",
    "   50 classes",
    "",
    "🔴 Limited Data",
    "   15 samples/class",
    "",
    "🔴 Curse of Dimensionality",
    "   d >> n problem",
]
ax3.text(0.1, 0.9, '\n'.join(challenges), transform=ax3.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8))

ax4 = fig.add_subplot(gs[1, :])
x = range(len(results_df))
colors_bars = ['#2ecc71' if k == 'linear' else '#e74c3c' for k in results_df['kernel']]
bars = ax4.bar(x, results_df['accuracy'], color=colors_bars, 
              edgecolor='black', linewidth=1.5)
ax4.axhline(y=results_df['accuracy'].mean(), color='blue', 
           linestyle='--', linewidth=2, label=f'Mean: {results_df["accuracy"].mean():.3f}')
ax4.set_xlabel('Configuration', fontweight='bold')
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Accuracy Across All Configurations', fontweight='bold', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels([f"{r['kernel'][:3].upper()}\nC={r['C']}" 
                     for _, r in results_df.iterrows()], fontsize=8)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[2, 0])
ax5.axis('tight')
ax5.axis('off')
dataset_info = [
    ['Property', 'Value'],
    ['Dataset', 'Amazon Review'],
    ['Train Samples', '750'],
    ['Test Samples', '750'],
    ['Features', '10,000'],
    ['Classes', '50'],
    ['Dimensionality', 'VERY HIGH'],
    ['Best Kernel', 'LINEAR'],
    ['Best Accuracy', f"{results_df['accuracy'].max():.4f}"],
]
table = ax5.table(cellText=dataset_info[1:], colLabels=dataset_info[0],
                 cellLoc='center', loc='center', colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)
table[(0, 0)].set_facecolor('#3498db')
table[(0, 1)].set_facecolor('#3498db')
table[(0, 0)].set_text_props(weight='bold', color='white')
table[(0, 1)].set_text_props(weight='bold', color='white')
for i in range(1, len(dataset_info)):
    if i % 2 == 0:
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 1)].set_facecolor('#f0f0f0')
ax5.set_title('Dataset Information', fontweight='bold', fontsize=11, pad=15)

ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
insights = [
    "KEY INSIGHTS:",
    "",
    "✓ Linear dominates:",
    "  58% vs 3% accuracy",
    "",
    "✓ RBF fails badly:",
    "  Curse of dimensionality",
    "",
    "✓ C parameter minimal",
    "  impact on Linear",
    "",
    "✓ Feature scaling",
    "  absolutely critical",
]
ax6.text(0.1, 0.9, '\n'.join(insights), transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.8))

ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('tight')
ax7.axis('off')
summary_stats = [
    ['Metric', 'Value'],
    ['Best Accuracy', f"{results_df['accuracy'].max():.4f}"],
    ['Worst Accuracy', f"{results_df['accuracy'].min():.4f}"],
    ['Linear Mean', f"{results_df[results_df['kernel']=='linear']['accuracy'].mean():.4f}"],
    ['RBF Mean', f"{results_df[results_df['kernel']=='rbf']['accuracy'].mean():.4f}"],
    ['Fastest Train', f"{results_df['train_time'].min():.3f}s"],
    ['Slowest Train', f"{results_df['train_time'].max():.3f}s"],
]
table2 = ax7.table(cellText=summary_stats[1:], colLabels=summary_stats[0],
                  cellLoc='center', loc='center', colWidths=[0.6, 0.4])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 2.2)
table2[(0, 0)].set_facecolor('#3498db')
table2[(0, 1)].set_facecolor('#3498db')
table2[(0, 0)].set_text_props(weight='bold', color='white')
table2[(0, 1)].set_text_props(weight='bold', color='white')
ax7.set_title('Summary Statistics', fontweight='bold', fontsize=11, pad=15)

plt.suptitle('Amazon Review: High-Dimensional SVM Classification Summary', 
            fontsize=16, fontweight='bold', y=0.98)

output_path = os.path.join(SCRIPT_DIR, 'amazon_viz_7_summary_dashboard.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()

best_idx = results_df['accuracy'].idxmax()
best_result = results_df.loc[best_idx]
worst_idx = results_df['accuracy'].idxmin()
worst_result = results_df.loc[worst_idx]
