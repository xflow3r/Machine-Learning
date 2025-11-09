#!/usr/bin/env python3
"""
Decision Tree Classifier for Amazon Review Dataset
Compares holdout validation vs. cross-validation with timing and accuracy metrics.
"""

import sys
from pathlib import Path
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Add parent directory to path to import preprocess_datasets
# Now 3 levels up: decision_tree -> reviews -> Exercise_1
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocess_datasets import load_amazon_review_dataset

# Random state for reproducibility
RANDOM_STATE = 2742

def prepare_data(x_train, y_train):
    """
    Prepare data by mapping ID column and dropping it for training.
    
    Returns:
        X_train_clean: DataFrame without ID column
        y_train: Series with target labels
    """
    # Drop ID column for training
    X_train_clean = x_train.drop(columns=['ID'])
    
    return X_train_clean, y_train


def train_holdout(X_train, y_train, holdout_pct=0.2, max_depth=None, min_samples_split=2):
    """
    Train decision tree using holdout validation.
    
    Args:
        X_train: Training features (without ID)
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
        X_train: Training features (without ID)
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


def predict_test(model, x_test):
    """
    Generate predictions on test set.
    
    Args:
        model: Trained classifier
        x_test: Test features (with ID column)
        
    Returns:
        DataFrame with ID and Class columns (matching original dataset format)
    """
    # Extract IDs
    test_ids = x_test['ID'].values
    
    # Drop ID for prediction
    X_test_clean = x_test.drop(columns=['ID'])
    
    # Predict
    predictions = model.predict(X_test_clean)
    
    # Create results DataFrame with columns matching original dataset (ID, Class)
    results_df = pd.DataFrame({
        'ID': test_ids,
        'Class': predictions
    })
    
    return results_df


def run_experiments():
    """
    Run decision tree experiments with different configurations.
    """
    print("Loading Amazon Review Dataset...")
    x_train, x_test, y_train, y_test = load_amazon_review_dataset(debug=False)
    print()
    
    # Prepare data
    print("Preparing data (dropping ID column)...")
    X_train_clean, y_train = prepare_data(x_train, y_train)
    print(f"Training data shape (without ID): {X_train_clean.shape}")
    print(f"Number of classes: {y_train.nunique()}")
    print()
    
    # Define parameter grid
    configs = [
        # Holdout with different splits
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.3, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'holdout', 'holdout_pct': 0.2, 'max_depth': 10, 'min_samples_split': 5},
        
        # Cross-validation with different folds
        {'method': 'cv', 'n_folds': 5, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'cv', 'n_folds': 10, 'max_depth': None, 'min_samples_split': 2},
        {'method': 'cv', 'n_folds': 5, 'max_depth': 10, 'min_samples_split': 5},
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
        
        results.append(result)
        
        print(f"  Method: {result['method']}")
        print(f"  Training time: {result['train_time']:.3f}s")
        print(f"  Train accuracy: {result['train_accuracy']:.4f}")
        print(f"  Val accuracy: {result['val_accuracy']:.4f}", end='')
        if 'val_accuracy_std' in result:
            print(f" (sd: {result['val_accuracy_std']:.4f})")
        else:
            print()
    
    print("\n" + "=" * 100)
    print("\nGenerating predictions on test set using best model...")
    
    # Use the model with best validation accuracy
    best_result = max(results, key=lambda r: r['val_accuracy'])
    print(f"Best model: {best_result['method']} with params {best_result['params']}")
    print(f"Best val accuracy: {best_result['val_accuracy']:.4f}")
    
    test_predictions = predict_test(best_result['model'], x_test)
    
    # Save predictions
    output_path = Path(__file__).parent / 'amazon_predictions.csv'
    test_predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Number of test predictions: {len(test_predictions)}")
    print(f"Columns: {list(test_predictions.columns)}")
    
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
            'Val Acc': f"{r['val_accuracy']:.4f}" + (f" (sd: {r['val_accuracy_std']:.4f})" if 'val_accuracy_std' in r else ''),
        }
        for r in results
    ])
    
    print(results_df.to_string(index=False))
    print("=" * 100)
    
    # Show confusion matrix for best model (if from holdout)
    if 'confusion_matrix' in best_result:
        print("\nConfusion Matrix (Best Model - Validation Set):")
        print(best_result['confusion_matrix'])
        
        # Calculate precision, recall, F1 for each class
        cm = best_result['confusion_matrix']
        n_classes = cm.shape[0]
        
        print("\nPer-Class Metrics (Best Model):")
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    print("\n" + "=" * 100)
    print("Experiments complete!")


if __name__ == '__main__':
    run_experiments()
