"""
Common utilities for decision tree experiments across all datasets.

This module provides shared functionality for:
- Random state for reproducibility
- Experiment configurations
- Training functions (holdout and cross-validation with GridSearchCV)
"""

import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Random state for reproducibility across all experiments
RANDOM_STATE = 2742


def get_cv_param_grid():
    """
    Return parameter grid for GridSearchCV hyperparameter tuning.
    
    Returns:
        Dictionary with parameter ranges for grid search
    """
    return {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20, 50],
        'min_samples_leaf': [1, 2, 5, 10]
    }


def get_holdout_experiment_configs():
    """
    Return holdout validation experiment configurations.
    
    Returns:
        List of dictionaries with holdout experiment configurations
    """
    return [
        # === Holdout validation with different splits and parameters ===
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': None, 'min_samples_split': 5},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': None, 'min_samples_split': 5},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 10, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': 10, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 10, 'min_samples_split': 5},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': 10, 'min_samples_split': 5},
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


def train_cross_validation(X_train, y_train, n_folds=5, param_grid=None):
    """
    Train a decision tree using GridSearchCV with k-fold cross-validation.
    Automatically finds the best hyperparameters from the parameter grid.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_folds: Number of cross-validation folds (default 5)
        param_grid: Parameter grid for GridSearchCV (default: uses get_cv_param_grid())
        
    Returns:
        Dictionary with best model, metrics, and selected parameters
    """
    start_time = time.time()
    
    # Use default parameter grid if none provided
    if param_grid is None:
        param_grid = get_cv_param_grid()
    
    # Create base classifier
    base_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=n_folds,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1
    )
    
    # Fit and find best parameters
    grid_search.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Get best model and its performance
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_  # Mean CV score for best params
    
    # Get train score for best model (retrain to get train accuracy)
    train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Get CV results for best params
    best_idx = grid_search.best_index_
    cv_results = grid_search.cv_results_
    val_std = cv_results['std_test_score'][best_idx]
    
    return {
        'model': best_model,
        'train_time': train_time,
        'train_accuracy': train_acc,
        'val_accuracy': best_score,
        'val_accuracy_std': val_std,
        'method': f'{n_folds}-Fold CV (GridSearchCV)',
        'params': best_params,
        'grid_search': grid_search  # Include full grid search object for detailed analysis
    }
