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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add parent directory to path to import preprocess_datasets
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocess_datasets import load_phishing_dataset

# Random state for reproducibility
RANDOM_STATE = 2742


def prepare_data(x_train, y_train, x_test):
    """
    Prepare data by dropping the Result_Label column (it's derived from target).

    Returns:
        X_train_clean, y_train, X_test_clean
    """
    if 'Result_Label' in x_train.columns:
        x_train = x_train.drop(columns=['Result_Label'])
    if 'Result_Label' in x_test.columns:
        x_test = x_test.drop(columns=['Result_Label'])
    return x_train, y_train, x_test


def train_holdout(X_train, y_train, holdout_pct=0.2, max_depth=None, min_samples_split=2):
    start_time = time.time()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=holdout_pct, random_state=RANDOM_STATE, stratify=y_train
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)

    train_time = time.time() - start_time

    y_pred_tr = clf.predict(X_tr)
    train_acc = accuracy_score(y_tr, y_pred_tr)

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
    start_time = time.time()

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE
    )

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
    print("Loading Phishing Dataset...")
    x_train, x_test, y_train, y_test = load_phishing_dataset(debug=False)
    print()

    print("Preparing data (dropping Result_Label column)...")
    X_train_clean, y_train, X_test_clean = prepare_data(x_train, y_train, x_test)
    print(f"Training data shape: {X_train_clean.shape}")
    print(f"Test data shape: {X_test_clean.shape}")
    print(f"Number of classes: {y_train.nunique()}")
    print(f"Target classes: {sorted(y_train.unique())} (-1=Legitimate, 0=Suspicious, 1=Phishing)")
    print()

    configs = [
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 10, 'min_samples_split': 5},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 5, 'min_samples_split': 10},
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
        else:  # cv
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
    best_result = max(results, key=lambda r: r['val_accuracy'])
    print(f"Best model: {best_result['method']} with params {best_result['params']}")
    print(f"Best val accuracy: {best_result['val_accuracy']:.4f}")
    print(f"Best test accuracy: {best_result['test_accuracy']:.4f}")

    # Print results summary table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Method':<20} {'Max Depth':<10} {'Min Split':<10} {'Train Time (s)':<15} {'Train Acc':<20} {'Val Acc':<8} {'Test Acc':<8}")

    for result in results:
        method = result['method']
        max_depth = result['params']['max_depth'] if result['params']['max_depth'] is not None else 'None'
        min_split = result['params']['min_samples_split']
        train_time = result['train_time']
        train_acc = result['train_accuracy']
        val_acc = result['val_accuracy']
        test_acc = result['test_accuracy']

        if 'val_accuracy_std' in result:
            val_acc_str = f"{val_acc:.4f} (sd: {result['val_accuracy_std']:.4f})"
        else:
            val_acc_str = f"{val_acc:.4f}"

        print(f"{method:<20} {str(max_depth):<10} {min_split:<10} {train_time:<15.3f} {train_acc:<20.4f} {val_acc_str:<8} {test_acc:<8.4f}")

    # Print confusion matrix and classification report for best model
    print("=" * 100)
    print("Confusion Matrix (Best Model - Test Set):")
    y_pred_test_best = best_result['model'].predict(X_test_clean)
    cm = confusion_matrix(y_test, y_pred_test_best)
    
    print("          Predicted")
    print("           -1    0    1")
    print("Actual")
    for i, label in enumerate([-1, 0, 1]):
        print(f"   {label:>2}    {cm[i][0]:>3}  {cm[i][1]:>3}  {cm[i][2]:>3}")
    
    print("\nPer-Class Metrics (Best Model - Test Set):")
    target_names = ['Legitimate (-1)', 'Suspicious (0)', 'Phishing (1)']
    report = classification_report(y_test, y_pred_test_best, target_names=target_names)
    print(report)
    
    # Calculate and print overall accuracy
    total_correct = cm.trace()
    total_samples = cm.sum()
    print(f"Overall Test Accuracy: {total_correct/total_samples:.4f} ({total_correct}/{total_samples})")

    print("\n" + "=" * 100)
    print("Experiments complete!")


if __name__ == '__main__':
    run_experiments()
