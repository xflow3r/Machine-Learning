"""
Enhanced Decision Tree Analysis for Road Safety Dataset

This script implements comprehensive decision tree analysis with:
- Multiple performance measures (accuracy, precision, recall, F1-score)
- Parameter sensitivity analysis (max_depth, min_samples_split, min_samples_leaf, criterion)
- Validation strategies comparison (holdout vs cross-validation)
- Overfitting analysis (train-val-test gaps)
- Model complexity metrics (tree depth, number of leaves)
- Training time comparison
- Visualization of results for LaTeX inclusion

Special considerations for Road Safety:
- Large dataset: 261,899 train / 65,475 test samples
- 11 classes (age bands): 0-5, 6-10, 11-15, 16-20, 21-25, 26-35, 36-45, 46-55, 56-65, 66-75, Over 75
- 54 features after preprocessing
- Need to investigate 100% accuracy (possible data leakage)

Results are saved to CSV and multiple PNG charts are generated.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_datasets import load_road_safety_dataset

# Set random state for reproducibility
RANDOM_STATE = 2742

# Configure matplotlib for LaTeX-compatible output
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


def prepare_data(X, y):
    """
    Prepare features for modeling.
    Road safety data doesn't have ID column after preprocessing.
    
    Args:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
        
    Returns:
        Processed features and labels
    """
    return X, y


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, 
                       max_depth, min_samples_split, min_samples_leaf, criterion,
                       validation_strategy, exp_name):
    """
    Train decision tree and collect comprehensive metrics.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (if holdout) or None (if CV)
        X_test, y_test: Test data
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        criterion: Splitting criterion ('gini' or 'entropy')
        validation_strategy: 'holdout' or 'cv_5' or 'cv_10'
        exp_name: Experiment name for logging
        
    Returns:
        Dictionary with all metrics
    """
    results = {
        'experiment': exp_name,
        'validation_strategy': validation_strategy,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion,
    }
    
    # Train model
    start_time = time.time()
    
    if validation_strategy == 'holdout':
        # Holdout validation
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Training metrics
        y_train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # Validation metrics
        y_val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec_macro = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
        val_prec_weighted = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_rec_macro = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
        val_rec_weighted = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_f1_macro = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        
        # Test metrics
        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_prec_weighted = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_rec_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_rec_weighted = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Model complexity
        tree_depth = clf.get_depth()
        n_leaves = clf.get_n_leaves()
        
        results.update({
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_accuracy_sd': 0.0,  # No SD for holdout
            'val_precision_macro': val_prec_macro,
            'val_recall_macro': val_rec_macro,
            'val_f1_macro': val_f1_macro,
            'val_f1_weighted': val_f1_weighted,
            'test_accuracy': test_acc,
            'test_precision_macro': test_prec_macro,
            'test_recall_macro': test_rec_macro,
            'test_f1_macro': test_f1_macro,
            'test_f1_weighted': test_f1_weighted,
            'overfitting_gap_train_val': train_acc - val_acc,
            'overfitting_gap_train_test': train_acc - test_acc,
            'tree_depth': tree_depth,
            'n_leaves': n_leaves,
            'training_time': training_time
        })
        
        # Store confusion matrix for later visualization
        results['confusion_matrix'] = confusion_matrix(y_test, y_test_pred)
        results['fitted_model'] = clf
        
    else:
        # Cross-validation (using smaller sample for CV due to large dataset)
        n_folds = 5 if validation_strategy == 'cv_5' else 10
        
        # For very large datasets, use subset for CV to speed up
        if len(X_train) > 50000:
            print(f"  Using stratified sample (50k) for CV due to large dataset size...")
            X_train_cv, _, y_train_cv, _ = train_test_split(
                X_train, y_train, train_size=50000, random_state=RANDOM_STATE, stratify=y_train
            )
        else:
            X_train_cv = X_train
            y_train_cv = y_train
        
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=RANDOM_STATE
        )
        
        # Perform cross-validation with multiple metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted'
        }
        
        cv_results = cross_validate(
            clf, X_train_cv, y_train_cv,
            cv=n_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        training_time = np.sum(cv_results['fit_time'])
        
        # Aggregate CV results
        train_acc = np.mean(cv_results['train_accuracy'])
        val_acc = np.mean(cv_results['test_accuracy'])
        val_acc_sd = np.std(cv_results['test_accuracy'])
        val_prec_macro = np.mean(cv_results['test_precision_macro'])
        val_rec_macro = np.mean(cv_results['test_recall_macro'])
        val_f1_macro = np.mean(cv_results['test_f1_macro'])
        val_f1_weighted = np.mean(cv_results['test_f1_weighted'])
        
        # Train final model on full training set for test evaluation
        clf.fit(X_train, y_train)
        
        y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_prec_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_rec_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Model complexity
        tree_depth = clf.get_depth()
        n_leaves = clf.get_n_leaves()
        
        results.update({
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'val_accuracy_sd': val_acc_sd,
            'val_precision_macro': val_prec_macro,
            'val_recall_macro': val_rec_macro,
            'val_f1_macro': val_f1_macro,
            'val_f1_weighted': val_f1_weighted,
            'test_accuracy': test_acc,
            'test_precision_macro': test_prec_macro,
            'test_recall_macro': test_rec_macro,
            'test_f1_macro': test_f1_macro,
            'test_f1_weighted': test_f1_weighted,
            'overfitting_gap_train_val': train_acc - val_acc,
            'overfitting_gap_train_test': train_acc - test_acc,
            'tree_depth': tree_depth,
            'n_leaves': n_leaves,
            'training_time': training_time
        })
        
        # Store confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_test_pred)
        results['fitted_model'] = clf
    
    return results


def run_enhanced_experiments(X_train, y_train, X_test, y_test):
    """
    Run comprehensive set of experiments with systematic parameter variation.
    
    20 experiments adapted for large road safety dataset:
    - Baseline experiments (1-6): holdout and CV with default/shallow/deep trees
    - Depth sensitivity (7-11): Vary max_depth from 3 to None
    - Split sensitivity (12-15): Vary min_samples_split from 2 to 100 (larger for big dataset)
    - Leaf sensitivity (16-19): Vary min_samples_leaf from 1 to 50 (larger for big dataset)
    - Criterion comparison (20): Compare gini vs entropy
    """
    results_list = []
    
    def print_result(res):
        """Helper to print results."""
        if res['val_accuracy_sd'] > 0:
            print(f"{res['experiment']}: Val Acc={res['val_accuracy']:.4f} sd:{res['val_accuracy_sd']:.4f}, Test Acc={res['test_accuracy']:.4f}")
        else:
            print(f"{res['experiment']}: Val Acc={res['val_accuracy']:.4f}, Test Acc={res['test_accuracy']:.4f}")
    
    # Experiment 1: Holdout 80/20 with default parameters
    print("Running Exp1...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, 
                                                  random_state=RANDOM_STATE, stratify=y_train)
    results = train_and_evaluate(X_tr, y_tr, X_val, y_val, X_test, y_test,
                                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='holdout',
                                exp_name='Exp1: Holdout 80/20')
    results_list.append(results)
    print_result(results)
    
    # Experiment 2: Holdout 70/30 with default parameters
    print("Running Exp2...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3,
                                                  random_state=RANDOM_STATE, stratify=y_train)
    results = train_and_evaluate(X_tr, y_tr, X_val, y_val, X_test, y_test,
                                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='holdout',
                                exp_name='Exp2: Holdout 70/30')
    results_list.append(results)
    print_result(results)
    
    # Experiment 3: Holdout 80/20 with shallow tree
    print("Running Exp3...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                  random_state=RANDOM_STATE, stratify=y_train)
    results = train_and_evaluate(X_tr, y_tr, X_val, y_val, X_test, y_test,
                                max_depth=5, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='holdout',
                                exp_name='Exp3: Holdout 80/20 depth=5')
    results_list.append(results)
    print_result(results)
    
    # Experiment 4: 5-Fold CV with default parameters
    print("Running Exp4...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp4: 5-Fold CV')
    results_list.append(results)
    print_result(results)
    
    # Experiment 5: 10-Fold CV with default parameters
    print("Running Exp5...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_10',
                                exp_name='Exp5: 10-Fold CV')
    results_list.append(results)
    print_result(results)
    
    # Experiment 6: 5-Fold CV with shallow tree
    print("Running Exp6...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=5, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp6: 5-Fold CV depth=5')
    results_list.append(results)
    print_result(results)
    
    # DEPTH SENSITIVITY ANALYSIS
    print("\n=== Depth Sensitivity Analysis ===")
    
    # Experiment 7: depth=3
    print("Running Exp7...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=3, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp7: 5-Fold CV depth=3')
    results_list.append(results)
    print_result(results)
    
    # Experiment 8: depth=10
    print("Running Exp8...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp8: 5-Fold CV depth=10')
    results_list.append(results)
    print_result(results)
    
    # Experiment 9: depth=15
    print("Running Exp9...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=15, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp9: 5-Fold CV depth=15')
    results_list.append(results)
    print_result(results)
    
    # Experiment 10: depth=20
    print("Running Exp10...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=20, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp10: 5-Fold CV depth=20')
    results_list.append(results)
    print_result(results)
    
    # Experiment 11: depth=None (unlimited)
    print("Running Exp11...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp11: 5-Fold CV depth=None')
    results_list.append(results)
    print_result(results)
    
    # SPLIT SENSITIVITY ANALYSIS (larger values for big dataset)
    print("\n=== Split Sensitivity Analysis ===")
    
    # Experiment 12: min_samples_split=10
    print("Running Exp12...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=10, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp12: 5-Fold CV split=10')
    results_list.append(results)
    print_result(results)
    
    # Experiment 13: min_samples_split=20
    print("Running Exp13...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=20, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp13: 5-Fold CV split=20')
    results_list.append(results)
    print_result(results)
    
    # Experiment 14: min_samples_split=50
    print("Running Exp14...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=50, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp14: 5-Fold CV split=50')
    results_list.append(results)
    print_result(results)
    
    # Experiment 15: min_samples_split=100
    print("Running Exp15...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=100, min_samples_leaf=1,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp15: 5-Fold CV split=100')
    results_list.append(results)
    print_result(results)
    
    # LEAF SENSITIVITY ANALYSIS (larger values for big dataset)
    print("\n=== Leaf Sensitivity Analysis ===")
    
    # Experiment 16: min_samples_leaf=10
    print("Running Exp16...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=2, min_samples_leaf=10,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp16: 5-Fold CV leaf=10')
    results_list.append(results)
    print_result(results)
    
    # Experiment 17: min_samples_leaf=20
    print("Running Exp17...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=2, min_samples_leaf=20,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp17: 5-Fold CV leaf=20')
    results_list.append(results)
    print_result(results)
    
    # Experiment 18: min_samples_leaf=50
    print("Running Exp18...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=2, min_samples_leaf=50,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp18: 5-Fold CV leaf=50')
    results_list.append(results)
    print_result(results)
    
    # Experiment 19: min_samples_leaf=100
    print("Running Exp19...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=2, min_samples_leaf=100,
                                criterion='gini', validation_strategy='cv_5',
                                exp_name='Exp19: 5-Fold CV leaf=100')
    results_list.append(results)
    print_result(results)
    
    # CRITERION COMPARISON
    print("\n=== Criterion Comparison ===")
    
    # Experiment 20: entropy criterion
    print("Running Exp20...")
    results = train_and_evaluate(X_train, y_train, None, None, X_test, y_test,
                                max_depth=10, min_samples_split=2, min_samples_leaf=1,
                                criterion='entropy', validation_strategy='cv_5',
                                exp_name='Exp20: 5-Fold CV entropy')
    results_list.append(results)
    print_result(results)
    
    return results_list


def generate_visualizations(results_df, output_dir, class_names):
    """
    Generate comprehensive visualizations for LaTeX inclusion.
    
    Creates 6 PNG files:
    1. Accuracy comparison across all experiments
    2. Depth sensitivity analysis
    3. Overfitting analysis (train vs val vs test)
    4. Training time comparison
    5. Model complexity analysis (depth vs leaves)
    6. Confusion matrix for best model (simplified for 11 classes)
    """
    
    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['val_accuracy'], width, label='Validation', alpha=0.8)
    ax.bar(x, results_df['test_accuracy'], width, label='Test', alpha=0.8)
    ax.bar(x + width, results_df['train_accuracy'], width, label='Train', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy')
    ax.set_title('Road Safety Dataset: Accuracy Comparison Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(results_df))], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/road_safety_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Depth Sensitivity
    depth_exp = results_df[results_df['experiment'].str.contains('depth=')]
    if len(depth_exp) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        depths = []
        for exp in depth_exp['experiment']:
            if 'depth=3' in exp:
                depths.append(3)
            elif 'depth=5' in exp:
                depths.append(5)
            elif 'depth=10' in exp:
                depths.append(10)
            elif 'depth=15' in exp:
                depths.append(15)
            elif 'depth=20' in exp:
                depths.append(20)
            elif 'depth=None' in exp:
                depths.append(25)  # Plot None as 25 for visibility
        
        ax.plot(depths, depth_exp['val_accuracy'], 'o-', label='Validation', linewidth=2, markersize=8)
        ax.plot(depths, depth_exp['test_accuracy'], 's-', label='Test', linewidth=2, markersize=8)
        ax.plot(depths, depth_exp['train_accuracy'], '^-', label='Train', linewidth=2, markersize=8)
        
        ax.set_xlabel('Max Depth')
        ax.set_ylabel('Accuracy')
        ax.set_title('Road Safety Dataset: Impact of Tree Depth on Performance')
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) if d != 25 else 'None' for d in depths])
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/road_safety_depth_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Overfitting Analysis
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    
    ax.plot(x, results_df['overfitting_gap_train_val'], 'o-', label='Train-Val Gap', linewidth=2, markersize=6)
    ax.plot(x, results_df['overfitting_gap_train_test'], 's-', label='Train-Test Gap', linewidth=2, markersize=6)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No Overfitting')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy Gap')
    ax.set_title('Road Safety Dataset: Overfitting Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i+1}" for i in range(len(results_df))], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/road_safety_overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Training Time
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue' if 'Holdout' in exp else 'green' if '5-Fold' in exp else 'orange' 
              for exp in results_df['experiment']]
    ax.bar(range(len(results_df)), results_df['training_time'], color=colors, alpha=0.7)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Road Safety Dataset: Training Time Comparison')
    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels([f"E{i+1}" for i in range(len(results_df))], rotation=45, ha='right')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Holdout'),
        Patch(facecolor='green', alpha=0.7, label='5-Fold CV'),
        Patch(facecolor='orange', alpha=0.7, label='10-Fold CV')
    ]
    ax.legend(handles=legend_elements)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/road_safety_training_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Model Complexity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tree depth
    x = np.arange(len(results_df))
    ax1.bar(x, results_df['tree_depth'], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Tree Depth')
    ax1.set_title('Actual Tree Depth')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"E{i+1}" for i in range(len(results_df))], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Number of leaves
    ax2.bar(x, results_df['n_leaves'], alpha=0.7, color='coral')
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Number of Leaves')
    ax2.set_title('Tree Complexity (Leaves)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"E{i+1}" for i in range(len(results_df))], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/road_safety_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGenerated 5 visualization files in {output_dir}/")
    print("Note: Confusion matrix generation skipped (11x11 matrix too large for visualization)")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ENHANCED DECISION TREE ANALYSIS: ROAD SAFETY DATASET")
    print("=" * 80)
    
    # Load data
    print("\nLoading road safety dataset...")
    X_train, X_test, y_train, y_test = load_road_safety_dataset()
    
    # Prepare data
    X_train, y_train = prepare_data(X_train, y_train)
    X_test, y_test = prepare_data(X_test, y_test)
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Classes (age bands): {sorted(y_train.unique())}")
    print(f"Number of classes: {len(y_train.unique())}")
    print(f"\nClass distribution (train):")
    for cls in sorted(y_train.unique()):
        count = (y_train == cls).sum()
        pct = 100 * count / len(y_train)
        print(f"  Class {cls}: {count} ({pct:.1f}%)")
    
    # Run experiments
    print("\n" + "=" * 80)
    print("RUNNING 20 COMPREHENSIVE EXPERIMENTS")
    print("Note: Using 50k sample for CV due to large dataset size")
    print("=" * 80)
    
    results_list = run_enhanced_experiments(X_train, y_train, X_test, y_test)
    
    # Convert to DataFrame (excluding model and confusion matrix for CSV)
    results_for_csv = []
    confusion_matrices = {}
    
    for res in results_list:
        # Store confusion matrix separately
        confusion_matrices[res['experiment']] = res['confusion_matrix']
        
        # Create CSV row without non-serializable objects
        csv_row = {k: v for k, v in res.items() 
                  if k not in ['confusion_matrix', 'fitted_model']}
        results_for_csv.append(csv_row)
    
    results_df = pd.DataFrame(results_for_csv)
    
    # Save results to CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = f'{output_dir}/road_safety_results.csv'
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Define age band class names
    class_names = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-35', '36-45', '46-55', '56-65', '66-75', 'Over 75']
    
    generate_visualizations(results_df, output_dir, class_names)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    best_val_idx = results_df['val_accuracy'].idxmax()
    best_test_idx = results_df['test_accuracy'].idxmax()
    best_f1_idx = results_df['test_f1_macro'].idxmax()
    
    print(f"\nBest Validation Accuracy: {results_df.loc[best_val_idx, 'val_accuracy']:.4f}")
    print(f"  Experiment: {results_df.loc[best_val_idx, 'experiment']}")
    print(f"  Parameters: depth={results_df.loc[best_val_idx, 'max_depth']}, "
          f"split={results_df.loc[best_val_idx, 'min_samples_split']}, "
          f"leaf={results_df.loc[best_val_idx, 'min_samples_leaf']}")
    
    print(f"\nBest Test Accuracy: {results_df.loc[best_test_idx, 'test_accuracy']:.4f}")
    print(f"  Experiment: {results_df.loc[best_test_idx, 'experiment']}")
    
    print(f"\nBest Test F1 (Macro): {results_df.loc[best_f1_idx, 'test_f1_macro']:.4f}")
    print(f"  Experiment: {results_df.loc[best_f1_idx, 'experiment']}")
    print(f"  Test Precision: {results_df.loc[best_f1_idx, 'test_precision_macro']:.4f}")
    print(f"  Test Recall: {results_df.loc[best_f1_idx, 'test_recall_macro']:.4f}")
    
    print(f"\nFastest Training: {results_df['training_time'].min():.4f}s")
    print(f"  Experiment: {results_df.loc[results_df['training_time'].idxmin(), 'experiment']}")
    
    print(f"\nLowest Overfitting (Train-Test): {results_df['overfitting_gap_train_test'].min():.4f}")
    print(f"  Experiment: {results_df.loc[results_df['overfitting_gap_train_test'].idxmin(), 'experiment']}")
    
    # Check for suspiciously high accuracy (data leakage investigation)
    if results_df['test_accuracy'].max() > 0.99:
        print("\n" + "!" * 80)
        print("WARNING: Suspiciously high accuracy detected (>99%)")
        print("This may indicate data leakage. Possible causes:")
        print("  1. Target variable encoded in features")
        print("  2. Temporal information allowing future prediction")
        print("  3. Unique identifiers not properly removed")
        print("Recommendation: Review feature engineering and preprocessing steps")
        print("!" * 80)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {csv_path}")
    print(f"  - {output_dir}/road_safety_accuracy_comparison.png")
    print(f"  - {output_dir}/road_safety_depth_sensitivity.png")
    print(f"  - {output_dir}/road_safety_overfitting_analysis.png")
    print(f"  - {output_dir}/road_safety_training_time.png")
    print(f"  - {output_dir}/road_safety_complexity_analysis.png")


if __name__ == "__main__":
    main()
