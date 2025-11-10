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


results_path = os.path.join(SCRIPT_DIR, 'voting_results.csv')
print(f"Loading results from: {results_path}")

if not os.path.exists(results_path):
    print(f"\n✗ ERROR: Results file not found!")
    print(f"  Please run 'voting_svm.py' first to generate results.")
    exit(1)

results_df = pd.read_csv(results_path)

results_df = results_df[results_df['accuracy'].notna()]

if len(results_df) == 0:
    print("\n✗ ERROR: No valid results found in CSV!")
    exit(1)


fig, ax = plt.subplots(figsize=(10, 6))

linear_results = results_df[results_df['kernel'] == 'linear']
rbf_results = results_df[results_df['kernel'] == 'rbf']

linear_best = linear_results['accuracy'].max() if len(linear_results) > 0 else 0
rbf_best = rbf_results['accuracy'].max() if len(rbf_results) > 0 else 0

kernels = ['Linear Kernel', 'RBF Kernel']
accuracies = [linear_best, rbf_best]
colors = ['#3498db', '#9b59b6']

bars = ax.bar(kernels, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.5)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}\n({height*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% Threshold')
ax.text(0.5, 0.96, '🎯 Excellent Performance Zone', 
        transform=ax.transData, ha='center', fontsize=11, 
        color='green', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

ax.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Voting Dataset: Linear vs RBF Performance\n(Binary Classification - Both Kernels Excel)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0.85, 1.0)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'voting_viz_1_kernel_comparison.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")

print("\n2. Creating parameter sensitivity plot...")

fig, ax = plt.subplots(figsize=(10, 6))

for kernel, color, marker, label in [
    ('linear', '#3498db', 'o', 'Linear'), 
    ('rbf', '#9b59b6', 's', 'RBF')
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
ax.set_title('Voting Dataset: Parameter Sensitivity\n(C=1.0 optimal for both kernels)', 
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.85, 1.0)

ax.text(0.02, 0.98, 
        'Key Finding:\n• Both kernels achieve >95%\n• C=1.0 gives best results\n• Small dataset favors simpler models',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'voting_viz_2_parameter_sensitivity.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()


fig, ax = plt.subplots(figsize=(10, 6))

time_by_kernel = results_df.groupby('kernel')['train_time'].agg(['mean', 'std']).reset_index()

colors_dict = {'linear': '#3498db', 'rbf': '#9b59b6'}
bar_colors = [colors_dict[k] for k in time_by_kernel['kernel']]

bars = ax.bar(time_by_kernel['kernel'], time_by_kernel['mean'], 
              yerr=time_by_kernel['std'], capsize=5,
              color=bar_colors, edgecolor='black', linewidth=2, alpha=0.8)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ms_time = height * 1000
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{ms_time:.2f}ms\n({height:.5f}s)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_xlabel('Kernel Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Voting Dataset: Training Time by Kernel\n(Extremely Fast - Small Dataset)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax.text(0.5, 0.95, 
        '⚡ Lightning Fast Training!\n(< 1 millisecond)',
        transform=ax.transAxes, fontsize=11, ha='center', verticalalignment='top',
        color='orange', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'voting_viz_3_training_time.png')
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
            vmin=0.85, vmax=1.0, cbar_kws={'label': 'Accuracy'},
            linewidths=2, linecolor='black', ax=ax, annot_kws={'fontsize': 11})

ax.set_title('Voting Dataset: SVM Performance Heatmap\n(All configurations achieve >85% accuracy)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('C Parameter', fontsize=12, fontweight='bold')
ax.set_ylabel('Kernel Type', fontsize=12, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'voting_viz_4_heatmap.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")

print("\n5. Creating comprehensive metrics table...")

fig, ax = plt.subplots(figsize=(15, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Kernel', 'C', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (ms)'])

for _, row in results_df.iterrows():
    table_data.append([
        row['kernel'].upper(),
        f"{row['C']:.1f}",
        f"{row['accuracy']:.4f}",
        f"{row['precision']:.4f}" if pd.notna(row['precision']) else 'N/A',
        f"{row['recall']:.4f}" if pd.notna(row['recall']) else 'N/A',
        f"{row['f1']:.4f}" if pd.notna(row['f1']) else 'N/A',
        f"{row['train_time']*1000:.2f}"  # Convert to ms
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
            table[(i, j)].set_facecolor('#f9e79f')
            table[(i, j)].set_text_props(weight='bold')

ax.set_title('Voting Dataset: Complete SVM Results (Binary Classification)\n(Best Result Highlighted in Gold)', 
            fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'voting_viz_5_metrics_table.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

for kernel, color in [('linear', '#3498db'), ('rbf', '#9b59b6')]:
    kernel_data = results_df[results_df['kernel'] == kernel].sort_values('C')
    ax1.plot(kernel_data['C'], kernel_data['accuracy']*100, 
            marker='o', label=kernel.upper(), linewidth=2, color=color, markersize=8)

ax1.set_xlabel('C Parameter', fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Accuracy vs C Parameter', fontweight='bold', fontsize=12)
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
ax1.set_ylim(85, 100)

kernel_f1 = results_df.groupby('kernel')['f1'].max().reset_index()
colors_list = ['#3498db' if k == 'linear' else '#9b59b6' for k in kernel_f1['kernel']]
bars = ax2.bar(kernel_f1['kernel'], kernel_f1['f1'], color=colors_list, 
              edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
ax2.set_ylabel('Best F1-Score', fontweight='bold')
ax2.set_title('F1-Score by Kernel', fontweight='bold', fontsize=12)
ax2.set_ylim(0.85, 1.0)
ax2.grid(True, alpha=0.3, axis='y')

ax3.scatter(results_df['train_time']*1000, results_df['accuracy']*100, 
           c=['#3498db' if k == 'linear' else '#9b59b6' for k in results_df['kernel']],
           s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
ax3.set_xlabel('Training Time (milliseconds)', fontweight='bold')
ax3.set_ylabel('Accuracy (%)', fontweight='bold')
ax3.set_title('Accuracy vs Training Time\n(Fast & Accurate!)', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(85, 100)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', label='Linear'),
                   Patch(facecolor='#9b59b6', label='RBF')]
ax3.legend(handles=legend_elements, loc='lower right')

ax4.axis('off')
advantages_text = """

"""

ax4.text(0.05, 0.95, advantages_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('Voting Dataset: Binary Classification Success Story', 
            fontsize=15, fontweight='bold')
plt.tight_layout()
output_path = os.path.join(SCRIPT_DIR, 'voting_viz_6_binary_classification.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_path}")

print("\n7. Creating comprehensive summary dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
kernel_best = results_df.groupby('kernel')['accuracy'].max().reset_index()
colors_list = ['#3498db' if k == 'linear' else '#9b59b6' for k in kernel_best['kernel']]
bars = ax1.bar(kernel_best['kernel'], kernel_best['accuracy'], 
              color=colors_list, edgecolor='black', linewidth=1.5)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}\n({height*100:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.set_ylabel('Best Accuracy', fontweight='bold')
ax1.set_title('Best Accuracy by Kernel', fontweight='bold', fontsize=11)
ax1.set_ylim(0.85, 1.0)
ax1.grid(True, alpha=0.3, axis='y')

ax2 = fig.add_subplot(gs[0, 1])
best_model = results_df.loc[results_df['accuracy'].idxmax()]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [best_model['accuracy'], best_model['precision'], 
          best_model['recall'], best_model['f1']]
bars = ax2.barh(metrics, values, color='#2ecc71', edgecolor='black', linewidth=1.5)
for bar in bars:
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2., f'{width:.3f}',
            ha='left', va='center', fontweight='bold', fontsize=10)
ax2.set_xlabel('Score', fontweight='bold')
ax2.set_title('Best Model: All Metrics', fontweight='bold', fontsize=11)
ax2.set_xlim(0.9, 1.0)
ax2.grid(True, alpha=0.3, axis='x')

ax3 = fig.add_subplot(gs[0, 2])
time_data = [results_df[results_df['kernel']=='linear']['train_time'].values * 1000,
             results_df[results_df['kernel']=='rbf']['train_time'].values * 1000]
bp = ax3.boxplot(time_data, labels=['Linear', 'RBF'], patch_artist=True)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#9b59b6')
ax3.set_ylabel('Training Time (milliseconds)', fontweight='bold')
ax3.set_title('Training Time Distribution', fontweight='bold', fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

ax4 = fig.add_subplot(gs[1, :])
x = range(len(results_df))
colors_bars = ['#3498db' if k == 'linear' else '#9b59b6' for k in results_df['kernel']]
bars = ax4.bar(x, results_df['accuracy'], color=colors_bars, 
              edgecolor='black', linewidth=1.5)
ax4.axhline(y=0.95, color='green', linestyle='--', linewidth=2, 
           label='95% Accuracy Threshold')
ax4.set_xlabel('Configuration', fontweight='bold')
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Accuracy Across All Configurations (Most >95%)', fontweight='bold', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels([f"{r['kernel'][:3].upper()}\nC={r['C']}" 
                     for _, r in results_df.iterrows()], fontsize=8)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0.85, 1.0)

ax5 = fig.add_subplot(gs[2, 0])
ax5.axis('tight')
ax5.axis('off')
dataset_info = [
    ['Property', 'Value'],
    ['Dataset', 'Voting Records'],
    ['Train Samples', '218'],
    ['Test Samples', '217'],
    ['Features', '16 (categorical)'],
    ['Classes', '2 (Democrat/Republican)'],
    ['Best Accuracy', f"{results_df['accuracy'].max():.4f}"],
    ['Best Kernel', results_df.loc[results_df['accuracy'].idxmax(), 'kernel'].upper()],
    ['Best C', f"{results_df.loc[results_df['accuracy'].idxmax(), 'C']:.1f}"],
]
table = ax5.table(cellText=dataset_info[1:], colLabels=dataset_info[0],
                 cellLoc='center', loc='center', colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.0)
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

]
ax6.text(0.1, 0.9, '\n'.join(insights), transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.8))


ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('tight')
ax7.axis('off')
summary_stats = [
    ['Metric', 'Value'],
    ['Total Experiments', f"{len(results_df)}"],
    ['Best Accuracy', f"{results_df['accuracy'].max():.4f}"],
    ['Worst Accuracy', f"{results_df['accuracy'].min():.4f}"],
    ['Mean Accuracy', f"{results_df['accuracy'].mean():.4f}"],
    ['Std Dev', f"{results_df['accuracy'].std():.4f}"],
    ['Fastest Training', f"{results_df['train_time'].min()*1000:.2f}ms"],
    ['Mean Training', f"{results_df['train_time'].mean()*1000:.2f}ms"],
]
table2 = ax7.table(cellText=summary_stats[1:], colLabels=summary_stats[0],
                  cellLoc='center', loc='center', colWidths=[0.6, 0.4])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 2.0)
table2[(0, 0)].set_facecolor('#3498db')
table2[(0, 1)].set_facecolor('#3498db')
table2[(0, 0)].set_text_props(weight='bold', color='white')
table2[(0, 1)].set_text_props(weight='bold', color='white')
ax7.set_title('Summary Statistics', fontweight='bold', fontsize=11, pad=15)

plt.suptitle('Voting Dataset: Binary SVM Classification - Complete Summary', 
            fontsize=16, fontweight='bold', y=0.98)

output_path = os.path.join(SCRIPT_DIR, 'voting_viz_7_summary_dashboard.png')
plt.savefig(output_path, bbox_inches='tight')
plt.close()


best_idx = results_df['accuracy'].idxmax()
best_result = results_df.loc[best_idx]


print("  Both Linear and RBF kernels achieve excellent results")
print("  Perfect example of SVM success on clean, structured data")
print("="*80)
