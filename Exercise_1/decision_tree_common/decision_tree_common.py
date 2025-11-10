"""
Common utilities for decision tree experiments across all datasets.

This module provides shared functionality for:
- Random state for reproducibility
- Experiment configurations
- Training functions (holdout and cross-validation with GridSearchCV)
- Visualization functions for saving results as PNG images
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
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


def save_table_as_image(df, output_path, title):
    """
    Save a pandas DataFrame as a PNG image.
    
    Args:
        df: DataFrame to visualize
        output_path: Path where to save the PNG
        title: Title for the table
    """
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='left', loc='center',
                     colWidths=[0.12] * len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Table saved to: {output_path}")


def save_classification_report_as_image(report_text, output_path, title):
    """
    Save a classification report as a PNG image.
    
    Args:
        report_text: Classification report text
        output_path: Path where to save the PNG
        title: Title for the report
    """
    # Parse classification report into DataFrame
    lines = report_text.strip().split('\n')
    data = []
    
    for line in lines[2:-3]:  # Skip header and footer lines
        parts = line.split()
        if len(parts) >= 5:
            data.append(parts)
    
    # Last 3 lines (accuracy, macro avg, weighted avg)
    for line in lines[-3:]:
        parts = line.split()
        if len(parts) >= 4:
            data.append(parts)
    
    if not data:
        return
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Classification report saved to: {output_path}")
