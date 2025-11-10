#!/usr/bin/env python3
"""
Decision Tree Classifier for Phishing Dataset
Compares holdout validation vs. cross-validation with timing and accuracy metrics.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path to import preprocess_datasets
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocess_datasets import load_phishing_dataset
from decision_tree_common.decision_tree_common import (
    get_holdout_experiment_configs,
    train_holdout,
    train_cross_validation
)


def prepare_data(x_train, x_test):
    """
    Prepare data by dropping the Result_Label column (it's derived from target).

    Returns:
        x_train, x_test - 'Result Label' column removed
    """
    if 'Result_Label' in x_train.columns:
        x_train = x_train.drop(columns=['Result_Label'])
    if 'Result_Label' in x_test.columns:
        x_test = x_test.drop(columns=['Result_Label'])
    return x_train, x_test


def run_experiments():
    """
    Run decision tree experiments with different configurations.
    """
    print("Loading Phishing Dataset...")
    x_train, x_test, y_train, y_test = load_phishing_dataset(debug=False)
    print()
    print("Preparing data (dropping Result_Label column)...")
    X_train_clean,  X_test_clean = prepare_data(x_train, x_test)
    print(f"Training data shape: {X_train_clean.shape}")
    print(f"Test data shape: {X_test_clean.shape}")
    print(f"Number of classes: {y_train.nunique()}")
    print(f"Target classes: {sorted(y_train.unique())} (-1=Legitimate, 0=Suspicious, 1=Phishing)")
    print()

    # Get holdout experiment configs
    holdout_configs = get_holdout_experiment_configs()
    
    results = []
    
    print("Running experiments...")
    print("=" * 100)
    
    # Run holdout experiments
    for i, config in enumerate(holdout_configs, 1):
        print(f"\nExperiment {i}/{len(holdout_configs)}: {config}")
        
        result = train_holdout(
            X_train_clean, y_train,
            holdout_pct=config['holdout_pct'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split']
        )

        # Evaluate on test set
        y_pred_test = result['model'].predict(X_test_clean)
        test_acc = accuracy_score(y_test, y_pred_test)
        result['test_accuracy'] = test_acc

        results.append(result)

        # Print summary
        print(f"  Method: {result['method']}")
        print(f"  Training time: {result['train_time']:.3f}s")
        print(f"  Train accuracy: {result['train_accuracy']:.4f}")
        print(f"  Val accuracy: {result['val_accuracy']:.4f}")
        print(f"  Test accuracy: {result['test_accuracy']:.4f}")

    # Run GridSearchCV experiment (automated hyperparameter tuning)
    print(f"\nExperiment {len(holdout_configs) + 1}/{len(holdout_configs) + 1}: GridSearchCV with 5-Fold CV")
    
    cv_result = train_cross_validation(X_train_clean, y_train, n_folds=5)
    
    # Evaluate on test set
    y_pred_test = cv_result['model'].predict(X_test_clean)
    test_acc = accuracy_score(y_test, y_pred_test)
    cv_result['test_accuracy'] = test_acc
    
    results.append(cv_result)
    
    # Print summary
    print(f"  Method: {cv_result['method']}")
    print(f"  Best params: {cv_result['params']}")
    print(f"  Training time: {cv_result['train_time']:.3f}s")
    print(f"  Train accuracy: {cv_result['train_accuracy']:.4f}")
    print(f"  Val accuracy: {cv_result['val_accuracy']:.4f} (sd: {cv_result['val_accuracy_std']:.4f})")
    print(f"  Test accuracy: {cv_result['test_accuracy']:.4f}")


    print("\n" + "=" * 100)
    print("Selecting best model based on validation accuracy...")
    
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
            'Max Depth': r['params']['max_depth'] if r['params']['max_depth'] is not None else 'None',
            'Min Split': r['params']['min_samples_split'],
            'Min Leaf': r['params'].get('min_samples_leaf', 'N/A'),
            'Train Time (s)': f"{r['train_time']:.3f}",
            'Train Acc': f"{r['train_accuracy']:.4f}",
            'Val Acc': f"{r['val_accuracy']:.4f}" + (f" (sd: {r['val_accuracy_std']:.4f})" if 'val_accuracy_std' in r else ''),
            'Test Acc': f"{r['test_accuracy']:.4f}",
        }
        for r in results
    ])
    
    print(results_df.to_string(index=False))
    print("=" * 100)
    
    # Show classification report from best model
    y_pred_test_best = best_result['model'].predict(X_test_clean)
    classification_rep = classification_report(y_test, y_pred_test_best)
    print("\nClassification Report (Best Model - Test Set):")
    print(classification_rep)

    print("\n" + "=" * 100)
    print("Experiments complete!")


if __name__ == '__main__':
    run_experiments()
