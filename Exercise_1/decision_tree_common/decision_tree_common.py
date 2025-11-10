"""
Common utilities for decision tree experiments across all datasets.

This module provides shared functionality for:
- Random state for reproducibility
- Experiment configurations
- Training functions (holdout and cross-validation)
"""

import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score

# Random state for reproducibility across all experiments
RANDOM_STATE = 2742


def get_experiment_configs():
    """
    Return standard experiment configurations for decision tree analysis.
    
    Returns:
        List of dictionaries with experiment configurations
    """
    return [
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


def train_holdout(X_train, y_train, holdout_pct=0.2, max_depth=None, min_samples_split=2):
    """
    Train a decision tree using holdout validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        holdout_pct: Percentage of data to use for validation (default 0.2)
        max_depth: Maximum depth of tree (default None = unlimited)
        min_samples_split: Minimum samples required to split (default 2)
        
    Returns:
        Dictionary with model, metrics, and configuration
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
    
    return {
        'model': clf,
        'train_time': train_time,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'method': f'Holdout ({int((1-holdout_pct)*100)}/{int(holdout_pct*100)})',
        'params': {'max_depth': max_depth, 'min_samples_split': min_samples_split}
    }


def train_cross_validation(X_train, y_train, n_folds=5, max_depth=None, min_samples_split=2):
    """
    Train a decision tree using k-fold cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_folds: Number of cross-validation folds (default 5)
        max_depth: Maximum depth of tree (default None = unlimited)
        min_samples_split: Minimum samples required to split (default 2)
        
    Returns:
        Dictionary with model, metrics, and configuration
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
