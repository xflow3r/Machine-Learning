#!/usr/bin/env python3
"""
Decision Tree Classifier for Road Safety Dataset
Compares holdout validation vs. cross-validation with timing and accuracy metrics.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add parent directory to path to import preprocess_datasets
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocess_datasets import load_road_safety_dataset

# Random state for reproducibility
RANDOM_STATE = 2742


def train_holdout(X_train, y_train, holdout_pct=0.2, max_depth=None, min_samples_split=2):
    """
    Train decision tree using holdout validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        holdout_pct: Percentage of data to use for validation
        max_depth: Max tree depth
        min_samples_split: Min samples required to split
        
    Returns:
        dict with model, metrics, and timing
    """
    start_time = time.time()
    
    # Split into train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=holdout_pct, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Train model
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)
    
    train_time = time.time() - start_time
    
    # Evaluate on training split
    y_pred_tr = clf.predict(X_tr)
    train_acc = accuracy_score(y_tr, y_pred_tr)
    
    # Evaluate on validation split
    y_pred_val = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    cm = confusion_matrix(y_val, y_pred_val)
    
    return {
        'model': clf,
        'train_time': train_time,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'confusion_matrix': cm,
        'method': f'Holdout ({int((1-holdout_pct)*100)}/{int(holdout_pct*100)})',
        'params': {'max_depth': max_depth, 'min_samples_split': min_samples_split}
    }


def train_cross_validation(X_train, y_train, n_folds=5, max_depth=None, min_samples_split=2):
    """
    Train decision tree using k-fold cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_folds: Number of CV folds
        max_depth: Max tree depth
        min_samples_split: Min samples required to split
        
    Returns:
        dict with model, metrics, and timing
    """
    start_time = time.time()
    
    # Create model
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE
    )
    
    # Perform cross-validation
    cv_results = cross_validate(
        clf, X_train, y_train,
        cv=n_folds,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1
    )
    
    # Train final model on full training data
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Compute mean CV scores
    mean_train_acc = cv_results['train_score'].mean()
    mean_val_acc = cv_results['test_score'].mean()
    std_val_acc = cv_results['test_score'].std()
    
    return {
        'model': clf,
        'train_time': train_time,
        'train_accuracy': mean_train_acc,
        'val_accuracy': mean_val_acc,
        'val_accuracy_std': std_val_acc,
        'method': f'{n_folds}-Fold CV',
        'params': {'max_depth': max_depth, 'min_samples_split': min_samples_split}
    }


def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict with test metrics
    """
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'test_accuracy': test_acc,
        'test_predictions': y_pred,
        'confusion_matrix': cm
    }


def run_experiments():
    """
    Run decision tree experiments with different configurations.
    """
    print("Loading Road Safety Dataset...")
    x_train, x_test, y_train, y_test = load_road_safety_dataset(debug=False)
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {y_train.nunique()}")
    print()
    
    # Define parameter grid
    configs = [
        # Holdout with different splits and parameters
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 10, 'min_samples_split': 5},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 15, 'min_samples_split': 10},
        
        # Cross-validation with different folds and parameters
        {'method': 'cv', 'n_folds': 5, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'cv', 'n_folds': 10, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 5},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 15, 'min_samples_split': 10},
    ]
    
    results = []
    
    print("Running experiments...")
    print("=" * 100)
    
    for i, config in enumerate(configs, 1):
        print(f"\nExperiment {i}/{len(configs)}: {config}")
        
        if config['method'] == 'holdout':
            result = train_holdout(
                x_train, y_train,
                holdout_pct=config['holdout_pct'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split']
            )
        else:  # CV
            result = train_cross_validation(
                x_train, y_train,
                n_folds=config['n_folds'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split']
            )
        
        # Evaluate on test set
        test_results = evaluate_on_test(result['model'], x_test, y_test)
        result['test_accuracy'] = test_results['test_accuracy']
        result['test_confusion_matrix'] = test_results['confusion_matrix']
        
        results.append(result)
        
        print(f"  Method: {result['method']}")
        print(f"  Training time: {result['train_time']:.3f}s")
        print(f"  Train accuracy: {result['train_accuracy']:.4f}")
        print(f"  Val accuracy: {result['val_accuracy']:.4f}", end='')
        if 'val_accuracy_std' in result:
            print(f" (sd: {result['val_accuracy_std']:.4f})")
        else:
            print()
        print(f"  Test accuracy: {result['test_accuracy']:.4f}")
    
    print("\n" + "=" * 100)
    print("\nSelecting best model based on validation accuracy...")
    
    # Use the model with best validation accuracy
    best_result = max(results, key=lambda r: r['val_accuracy'])
    print(f"Best model: {best_result['method']} with params {best_result['params']}")
    print(f"Best val accuracy: {best_result['val_accuracy']:.4f}")
    print(f"Best test accuracy: {best_result['test_accuracy']:.4f}")
    
    # Create results summary table
    print("\n" + "=" * 100)
    print("\nRESULTS SUMMARY TABLE")
    print("=" * 100)
    
    results_df = pd.DataFrame([
        {
            'Method': r['method'],
            'Max Depth': r['params']['max_depth'] if r['params']['max_depth'] else 'None',
            'Min Split': r['params']['min_samples_split'],
            'Train Time (s)': f"{r['train_time']:.3f}",
            'Train Acc': f"{r['train_accuracy']:.4f}",
            'Val Acc': f"{r['val_accuracy']:.4f}" + (f" (±{r['val_accuracy_std']:.4f})" if 'val_accuracy_std' in r else ''),
            'Test Acc': f"{r['test_accuracy']:.4f}",
        }
        for r in results
    ])
    
    print(results_df.to_string(index=False))
    print("=" * 100)
    
    # Show confusion matrix for best model
    print("\nConfusion Matrix (Best Model - Test Set):")
    print(best_result['test_confusion_matrix'])
    
    # Calculate per-class metrics for best model
    cm = best_result['test_confusion_matrix']
    n_classes = cm.shape[0]
    
    print(f"\nPer-Class Metrics (Best Model - Test Set):")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{i+1:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    # Overall metrics
    total_correct = np.trace(cm)
    total_samples = cm.sum()
    overall_acc = total_correct / total_samples
    
    print("-" * 60)
    print(f"Overall Test Accuracy: {overall_acc:.4f} ({total_correct}/{int(total_samples)})")
    
    print("\n" + "=" * 100)
    print("Experiments complete!")


if __name__ == '__main__':
    run_experiments()
