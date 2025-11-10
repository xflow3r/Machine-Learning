#!/usr/bin/env python3
"""
Decision Tree Classifier for Phishing Dataset
Compares holdout validation vs. cross-validation with timing and accuracy metrics.
"""

import sys
from pathlib import Path
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path to import preprocess_datasets
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocess_datasets import load_phishing_dataset

# Random state for reproducibility
RANDOM_STATE = 2742


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


def train_holdout(X_train, y_train, holdout_pct=0.2, max_depth=None, min_samples_split=2):
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
    
    return {
        'model': clf,
        'train_time': train_time,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'method': f'Holdout ({int((1-holdout_pct)*100)}/{int(holdout_pct*100)})',
        'params': {'max_depth': max_depth, 'min_samples_split': min_samples_split}
    }


def train_cross_validation(X_train, y_train, n_folds=5, max_depth=None, min_samples_split=2):
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

    # Fit final model on full training set
    clf.fit(X_train, y_train)

    train_time = time.time() - start_time

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

    configs = [
        # Holdout with different splits and parameters
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 10, 'min_samples_split': 5},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 5, 'min_samples_split': 10},

        # Cross-validation with different folds and parameters
        {'method': 'cv', 'n_folds': 5, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'cv', 'n_folds': 10, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 5},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 5, 'min_samples_split': 10},
    ]
    
    results = []
    
    print("Running experiments...")
    print("=" * 100)
    
    for i, config in enumerate(configs, 1):
        print(f"\nExperiment {i}/{len(configs)}: {config}")
        
        if config['method'] == 'holdout':
            result = train_holdout(
                X_train_clean, y_train,
                holdout_pct=config['holdout_pct'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split']
            )
        else:  # CV
            result = train_cross_validation(
                X_train_clean, y_train,
                n_folds=config['n_folds'],
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
        if 'val_accuracy_std' in result:
            print(f"  Val accuracy: {result['val_accuracy']:.4f} (sd: {result['val_accuracy_std']:.4f})")
        else:
            print(f"  Val accuracy: {result['val_accuracy']:.4f}")
        print(f"  Test accuracy: {result['test_accuracy']:.4f}")

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
