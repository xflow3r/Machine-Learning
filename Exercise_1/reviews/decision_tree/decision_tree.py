#!/usr/bin/env python3
"""
Decision Tree Classifier for Amazon Review Dataset
Compares holdout validation vs. cross-validation with timing and accuracy metrics.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

# Add parent directory to path to import preprocess_datasets
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from preprocess_datasets import load_amazon_review_dataset
from decision_tree_common.decision_tree_common import (
    get_holdout_experiment_configs,
    train_holdout,
    train_cross_validation
)


def prepare_data(x_train):
    """
    Prepare data by dropping the ID column before training.
    
    Returns:
        X_train_clean: DataFrame without ID column for training
    """
    X_train_clean = x_train.drop(columns=['ID'])

    return X_train_clean


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
    print("Preparing data (dropping ID column)...")
    X_train_clean = prepare_data(x_train)
    print(f"Training data shape (without ID column): {X_train_clean.shape}")
    print(f"Number of classes: {y_train.nunique()}")
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
        
        # test set has no labels, so we skip test accuracy calculation
        results.append(result)

        # Print summary
        print(f"  Method: {result['method']}")
        print(f"  Training time: {result['train_time']:.3f}s")
        print(f"  Train accuracy: {result['train_accuracy']:.4f}")
        print(f"  Val accuracy: {result['val_accuracy']:.4f}")

    # Run GridSearchCV experiments with different fold counts
    cv_folds = [5, 10]
    for fold_idx, n_folds in enumerate(cv_folds, 1):
        exp_num = len(holdout_configs) + fold_idx
        total_exps = len(holdout_configs) + len(cv_folds)
        print(f"\nExperiment {exp_num}/{total_exps}: GridSearchCV with {n_folds}-Fold CV")
        
        cv_result = train_cross_validation(X_train_clean, y_train, n_folds=n_folds)
        
        results.append(cv_result)
        
        # Print summary
        print(f"  Method: {cv_result['method']}")
        print(f"  Best params: {cv_result['params']}")
        print(f"  Training time: {cv_result['train_time']:.3f}s")
        print(f"  Train accuracy: {cv_result['train_accuracy']:.4f}")
        print(f"  Val accuracy: {cv_result['val_accuracy']:.4f} (sd: {cv_result['val_accuracy_std']:.4f})")


    print("\n" + "=" * 100)
    print("Selecting best model based on validation accuracy...")
    
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
            'Max Depth': r['params']['max_depth'] if r['params']['max_depth'] is not None else 'None',
            'Min Split': r['params']['min_samples_split'],
            'Min Leaf': r['params'].get('min_samples_leaf', 'N/A'),
            'Train Time (s)': f"{r['train_time']:.3f}",
            'Train Acc': f"{r['train_accuracy']:.4f}",
            'Val Acc': f"{r['val_accuracy']:.4f}" + (f" (sd: {r['val_accuracy_std']:.4f})" if 'val_accuracy_std' in r else ''),
        }
        for r in results
    ])
    
    print(results_df.to_string(index=False))
    print("=" * 100)
    
    # Show classification report from best model (if from holdout)
    if best_result['method'].startswith('Holdout'):
        y_pred_val_best = best_result['model'].predict(X_train_clean)
        classification_rep = classification_report(y_train, y_pred_val_best)
        print("\nClassification Report (Best Model - Validation Set):")
        print(classification_rep)
    else:
        print('No confusion matrix available, best model was CV-based.')

    print("\n" + "=" * 100)
    print("Experiments complete!")


if __name__ == '__main__':
    run_experiments()
