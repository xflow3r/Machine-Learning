#!/usr/bin/env python3
"""
Enhanced Decision Tree Classifier for Phishing Dataset
Comprehensive analysis with multiple performance measures, parameter sensitivity, and visualization.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocess_datasets import load_phishing_dataset

# Random state for reproducibility
RANDOM_STATE = 2742

# Configure matplotlib for LaTeX-compatible output
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300


def prepare_data(x_train, y_train, x_test):
    """Drop Result_Label column (derived from target)."""
    if 'Result_Label' in x_train.columns:
        x_train = x_train.drop(columns=['Result_Label'])
    if 'Result_Label' in x_test.columns:
        x_test = x_test.drop(columns=['Result_Label'])
    return x_train, y_train, x_test


def train_and_evaluate(X_train, y_train, X_test, y_test, config):
    """Train model and return comprehensive metrics."""
    method = config['method']
    start_time = time.time()
    
    # Create model
    clf = DecisionTreeClassifier(
        max_depth=config.get('max_depth'),
        min_samples_split=config.get('min_samples_split', 2),
        min_samples_leaf=config.get('min_samples_leaf', 1),
        criterion=config.get('criterion', 'gini'),
        random_state=RANDOM_STATE
    )
    
    if method == 'holdout':
        # Holdout validation
        holdout_pct = config.get('holdout_pct', 0.2)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=holdout_pct, 
            random_state=RANDOM_STATE, stratify=y_train
        )
        
        clf.fit(X_tr, y_tr)
        train_time = time.time() - start_time
        
        # Training metrics
        y_pred_tr = clf.predict(X_tr)
        train_acc = accuracy_score(y_tr, y_pred_tr)
        
        # Validation metrics
        y_pred_val = clf.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred_val)
        
        # Macro/weighted metrics
        _, _, _, _ = precision_recall_fscore_support(y_val, y_pred_val, average=None, zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_val, y_pred_val, average='macro', zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_val, y_pred_val, average='weighted', zero_division=0
        )
        
        val_std = 0.0
        method_name = f'Holdout ({int((1-holdout_pct)*100)}/{int(holdout_pct*100)})'
        
    else:  # Cross-validation
        n_folds = config.get('n_folds', 5)
        
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted',
        }
        
        cv_results = cross_validate(
            clf, X_train, y_train,
            cv=n_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        train_acc = cv_results['train_accuracy'].mean()
        val_acc = cv_results['test_accuracy'].mean()
        val_std = cv_results['test_accuracy'].std()
        macro_p = cv_results['test_precision_macro'].mean()
        macro_r = cv_results['test_recall_macro'].mean()
        macro_f1 = cv_results['test_f1_macro'].mean()
        weighted_p = cv_results['test_precision_weighted'].mean()
        weighted_r = cv_results['test_recall_weighted'].mean()
        weighted_f1 = cv_results['test_f1_weighted'].mean()
        
        method_name = f'{n_folds}-Fold CV'
    
    # Test evaluation
    y_pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_macro_p, test_macro_r, test_macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='macro', zero_division=0
    )
    
    # Model complexity
    tree_depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()
    
    # Overfitting metrics
    overfit_gap = train_acc - val_acc
    train_test_gap = train_acc - test_acc
    
    return {
        'method': method_name,
        'max_depth': config.get('max_depth'),
        'min_samples_split': config.get('min_samples_split', 2),
        'min_samples_leaf': config.get('min_samples_leaf', 1),
        'criterion': config.get('criterion', 'gini'),
        'train_time': train_time,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_std': val_std,
        'test_accuracy': test_acc,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_p,
        'weighted_recall': weighted_r,
        'weighted_f1': weighted_f1,
        'test_macro_f1': test_macro_f1,
        'tree_depth': tree_depth,
        'n_leaves': n_leaves,
        'overfit_gap': overfit_gap,
        'train_test_gap': train_test_gap,
        'model': clf,
    }


def generate_visualizations(results_df, output_dir):
    """Generate PNG charts for LaTeX inclusion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Accuracy comparison across methods
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['train_accuracy'], width, label='Train', alpha=0.8)
    ax.bar(x, results_df['val_accuracy'], width, label='Validation', alpha=0.8)
    ax.bar(x + width, results_df['test_accuracy'], width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy')
    ax.set_title('Decision Tree Accuracy Comparison (Phishing Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(results_df)+1))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'phishing_accuracy_comparison.png', bbox_inches='tight')
    plt.close()
    
    # 2. Parameter sensitivity: max_depth vs accuracy
    depth_data = results_df[results_df['max_depth'].notna()].copy()
    if len(depth_data) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in depth_data['method'].unique():
            method_data = depth_data[depth_data['method'] == method]
            ax.plot(method_data['max_depth'], method_data['val_accuracy'], 
                   marker='o', label=method, linewidth=2)
        
        ax.set_xlabel('Max Depth')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Parameter Sensitivity: Max Depth vs Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'phishing_depth_sensitivity.png', bbox_inches='tight')
        plt.close()
    
    # 3. Overfitting analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    
    ax.bar(x, results_df['overfit_gap'], label='Train-Val Gap', alpha=0.7)
    ax.bar(x, results_df['train_test_gap'], label='Train-Test Gap', alpha=0.7)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy Gap')
    ax.set_title('Overfitting Analysis (Phishing Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(results_df)+1))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'phishing_overfitting_analysis.png', bbox_inches='tight')
    plt.close()
    
    # 4. Training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4' if 'Holdout' in m else '#ff7f0e' for m in results_df['method']]
    ax.bar(range(len(results_df)), results_df['train_time'], color=colors, alpha=0.7)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison (Phishing Dataset)')
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(range(1, len(results_df)+1))
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Holdout'),
        Patch(facecolor='#ff7f0e', label='Cross-Validation')
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(output_dir / 'phishing_training_time.png', bbox_inches='tight')
    plt.close()
    
    # 5. Model complexity vs performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(results_df['tree_depth'], results_df['val_accuracy'], 
               s=100, alpha=0.6, c=results_df['train_time'], cmap='viridis')
    ax1.set_xlabel('Tree Depth')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Tree Depth vs Accuracy')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(results_df['n_leaves'], results_df['val_accuracy'], 
               s=100, alpha=0.6, c=results_df['train_time'], cmap='viridis')
    ax2.set_xlabel('Number of Leaves')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Model Complexity vs Accuracy')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phishing_complexity_analysis.png', bbox_inches='tight')
    plt.close()
    
    # 6. Confusion matrix heatmap for best model
    best_idx = results_df['val_accuracy'].idxmax()
    best_model = results_df.loc[best_idx, 'model']
    
    X_train, X_test, y_train, y_test = load_phishing_dataset(debug=False)
    X_train, y_train, X_test = prepare_data(X_train, y_train, X_test)
    
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Suspicious', 'Phishing'],
                yticklabels=['Legitimate', 'Suspicious', 'Phishing'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - Best Model (Experiment {best_idx+1})')
    plt.tight_layout()
    plt.savefig(output_dir / 'phishing_confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Generated 6 visualization PNG files in {output_dir}/")


def run_enhanced_experiments():
    """Run comprehensive experiments with expanded parameter grid."""
    print("="*100)
    print("ENHANCED DECISION TREE ANALYSIS - PHISHING DATASET")
    print("="*100)
    
    # Load data
    print("\nLoading and preparing data...")
    X_train, X_test, y_train, y_test = load_phishing_dataset(debug=False)
    X_train, y_train, X_test = prepare_data(X_train, y_train, X_test)
    
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Classes: {sorted(y_train.unique())} (-1=Legitimate, 0=Suspicious, 1=Phishing)")
    
    # Expanded parameter grid for comprehensive analysis
    configs = [
        # Baseline: different validation methods
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 10, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        
        # Parameter sensitivity: max_depth (single parameter variation)
        {'method': 'cv', 'n_folds': 5, 'max_depth': 3, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        
        # Parameter sensitivity: min_samples_split (single parameter variation)
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 20, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 50, 'min_samples_leaf': 1, 'criterion': 'gini'},
        
        # Parameter sensitivity: min_samples_leaf (single parameter variation)
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 5, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 10, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 20, 'criterion': 'gini'},
        
        # Parameter sensitivity: criterion (single parameter variation)
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy'},
        
        # Combinations to test interactions
        {'method': 'cv', 'n_folds': 5, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'criterion': 'gini'},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'criterion': 'entropy'},
    ]
    
    results = []
    print(f"\nRunning {len(configs)} experiments...")
    print("="*100)
    
    for i, config in enumerate(configs, 1):
        print(f"\nExperiment {i}/{len(configs)}: {config['method']}, depth={config.get('max_depth')}, "
              f"split={config.get('min_samples_split')}, leaf={config.get('min_samples_leaf')}, "
              f"criterion={config.get('criterion')}")
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, config)
        results.append(result)
        
        print(f"  Train: {result['train_accuracy']:.4f} | Val: {result['val_accuracy']:.4f} | "
              f"Test: {result['test_accuracy']:.4f} | Time: {result['train_time']:.3f}s")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_dir = Path(__file__).parent
    csv_path = output_dir / 'phishing_results.csv'
    results_df_save = results_df.drop(columns=['model'])  # Don't save model objects
    results_df_save.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(results_df, output_dir)
    
    # Print summary
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    summary_cols = ['method', 'max_depth', 'min_samples_split', 'criterion', 
                   'train_accuracy', 'val_accuracy', 'test_accuracy', 'macro_f1', 
                   'overfit_gap', 'train_time']
    
    summary_df = results_df[summary_cols].copy()
    summary_df['max_depth'] = summary_df['max_depth'].fillna('None')
    summary_df = summary_df.round(4)
    
    print(summary_df.to_string(index=True))
    
    # Best model analysis
    print("\n" + "="*100)
    print("BEST MODEL ANALYSIS")
    print("="*100)
    
    best_idx = results_df['val_accuracy'].idxmax()
    best = results_df.loc[best_idx]
    
    print(f"\nBest Model (Experiment {best_idx+1}):")
    print(f"  Method: {best['method']}")
    print(f"  Parameters: depth={best['max_depth']}, split={best['min_samples_split']}, "
          f"leaf={best['min_samples_leaf']}, criterion={best['criterion']}")
    print(f"  Validation Accuracy: {best['val_accuracy']:.4f}")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
    print(f"  Macro F1: {best['macro_f1']:.4f}")
    print(f"  Weighted F1: {best['weighted_f1']:.4f}")
    print(f"  Overfitting Gap: {best['overfit_gap']:.4f}")
    print(f"  Tree Depth: {best['tree_depth']}")
    print(f"  Number of Leaves: {best['n_leaves']}")
    print(f"  Training Time: {best['train_time']:.3f}s")
    
    print("\n" + "="*100)
    print("Analysis complete! Check the generated PNG files for visualizations.")
    print("="*100)


if __name__ == '__main__':
    run_enhanced_experiments()
