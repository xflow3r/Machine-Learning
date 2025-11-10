"""
SVM Classification for Amazon Review Dataset
Exercise 1 - Machine Learning
Dataset: Amazon Review (CSV format)
- 750 training samples, 10,001 features (high-dimensional), 50 classes
- Kaggle competition dataset
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '../../Datasets')
TRAIN_PATH = os.path.join(DATASET_DIR, 'amazon_review_learn.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'amazon_review_test.csv')

print("="*80)
print("AMAZON REVIEW DATASET - SVM CLASSIFICATION")
print("="*80)
print(f"Script directory: {SCRIPT_DIR}")
print(f"Training data: {TRAIN_PATH}")
print(f"Test data: {TEST_PATH}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_amazon_review_dataset():
    """
    Load the Amazon Review dataset (CSV format)
    Returns: X_train, y_train, X_test, test_ids, train_ids
    """
    print("\n" + "="*80)
    print("LOADING AMAZON REVIEW DATASET")
    print("="*80)
    
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Cannot find training data: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Cannot find test data: {TEST_PATH}")
    
    # Load training data
    print(f"Loading training data from: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    
    # Load test data
    print(f"Loading test data from: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    
    # Extract features and target
    # Last column is the target (class)
    y_train = train_df.iloc[:, -1]
    X_train = train_df.iloc[:, :-1]
    
    # Store IDs before dropping
    train_ids = X_train['ID'].values if 'ID' in X_train.columns else None
    test_ids = test_df['ID'].values if 'ID' in test_df.columns else None
    
    # Drop ID column from features
    if 'ID' in X_train.columns:
        X_train = X_train.drop('ID', axis=1)
    if 'ID' in test_df.columns:
        X_test = test_df.drop('ID', axis=1)
    else:
        X_test = test_df.copy()
    
    print(f"✓ Training data: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"✓ Test data: {len(X_test)} samples, {X_test.shape[1]} features")
    print(f"  Number of classes: {len(y_train.unique())}")
    print(f"  Class distribution (top 10):")
    for cls, count in list(y_train.value_counts().items())[:10]:
        print(f"    {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Check for missing values
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    if missing_train > 0:
        print(f"  ⚠ Training data has {missing_train} missing values")
    if missing_test > 0:
        print(f"  ⚠ Test data has {missing_test} missing values")
    
    return X_train, y_train, X_test, test_ids, train_ids


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_amazon_data(X_train, X_test):
    """
    Preprocess Amazon Review dataset:
    - Handle missing values
    - Scale features (CRITICAL for high-dimensional SVM!)
    
    Returns: X_train_scaled, X_test_scaled, scaler
    """
    print("\n" + "="*80)
    print("PREPROCESSING")
    print("="*80)
    
    # Make copies
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Handle missing values (fill with column mean)
    print("  Handling missing values...")
    for col in X_train.columns:
        col_mean = X_train[col].mean()
        if pd.isna(col_mean):
            col_mean = 0
        X_train[col] = X_train[col].fillna(col_mean)
        
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(col_mean)
    
    # Scale features (CRITICAL for SVM, especially high-dimensional!)
    print("  Scaling features (StandardScaler)...")
    print("  ⚠ This is CRITICAL for SVM performance on high-dimensional data!")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Preprocessing complete")
    print(f"  Training shape: {X_train_scaled.shape}")
    print(f"  Test shape: {X_test_scaled.shape}")
    print(f"  Feature scaling: mean=0, std=1")
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# SVM PARAMETER GRID
# ============================================================================

def get_svm_param_grid():
    """
    Define SVM parameter combinations to test
    For high-dimensional data, Linear kernel is usually better
    """
    params = [
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'linear', 'C': 1.0},
        {'kernel': 'linear', 'C': 10.0},
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
    ]
    return params


# ============================================================================
# TRAIN AND EVALUATE SVM
# ============================================================================

def train_and_evaluate_svm(X_train, X_val, y_train, y_val, params):
    """
    Train SVM with given parameters and evaluate
    Returns: dictionary with results
    """
    print(f"\n  Testing SVM with {params}...")
    
    result = {
        'params': str(params),
        'kernel': params['kernel'],
        'C': params['C'],
    }
    
    try:
        # Create and train SVM
        svm = SVC(**params, random_state=42)
        
        start_time = time.time()
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict on validation set
        y_pred = svm.predict(X_val)
        
        # Metrics
        result['accuracy'] = accuracy_score(y_val, y_pred)
        result['train_time'] = train_time
        
        # Additional metrics (weighted for multi-class)
        try:
            result['precision'] = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            result['recall'] = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            result['f1'] = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        except:
            result['precision'] = np.nan
            result['recall'] = np.nan
            result['f1'] = np.nan
        
        print(f"    ✓ Accuracy: {result['accuracy']:.4f}, Time: {train_time:.2f}s")
        
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        result['error'] = str(e)
        result['accuracy'] = np.nan
        result['train_time'] = np.nan
        result['precision'] = np.nan
        result['recall'] = np.nan
        result['f1'] = np.nan
    
    return result


# ============================================================================
# CREATE KAGGLE SUBMISSION
# ============================================================================

def create_kaggle_submission(X_train_full, y_train_full, X_test, test_ids, best_params):
    """
    Train final model on ALL training data and create Kaggle submission
    """
    print("\n" + "="*80)
    print("CREATING KAGGLE SUBMISSION")
    print("="*80)
    print(f"Training on full dataset: {len(X_train_full)} samples")
    print(f"Using best parameters: {best_params}")
    
    # Preprocess full data
    X_train_prep, X_test_prep, scaler = preprocess_amazon_data(
        pd.DataFrame(X_train_full), pd.DataFrame(X_test)
    )
    
    # Train final model
    print("  Training final SVM model...")
    svm = SVC(**best_params, random_state=42)
    start_time = time.time()
    svm.fit(X_train_prep, y_train_full)
    train_time = time.time() - start_time
    print(f"  ✓ Training complete in {train_time:.2f}s")
    
    # Predict on test set
    print("  Making predictions on test set...")
    y_pred = svm.predict(X_test_prep)
    print(f"  ✓ Generated {len(y_pred)} predictions")
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'ID': test_ids,
        'class': y_pred
    })
    
    # Save submission
    output_path = os.path.join(SCRIPT_DIR, 'kaggle_amazon_review.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Kaggle submission saved: {output_path}")
    print(f"  Format: ID, class")
    print(f"  Rows: {len(submission)}")
    print(f"  Ready to upload to Kaggle!")
    
    return output_path


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_amazon_review_experiments():
    """
    Main function to run all SVM experiments on Amazon Review dataset
    """
    print("\n" + "="*80)
    print("STARTING SVM EXPERIMENTS")
    print("="*80)
    
    # Load dataset
    X_train_full, y_train_full, X_test_full, test_ids, train_ids = load_amazon_review_dataset()
    
    # Create train/validation split for experiments (80/20)
    print("\n" + "="*80)
    print("CREATING TRAIN/VALIDATION SPLIT")
    print("="*80)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    
    # Preprocess
    X_train_prep, X_val_prep, scaler = preprocess_amazon_data(X_train, X_val)
    
    # Get parameter grid
    params_grid = get_svm_param_grid()
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    print("NOTE: High-dimensional data (10,000 features)")
    print("      Linear kernel typically performs better than RBF")
    
    all_results = []
    
    for params in params_grid:
        result = train_and_evaluate_svm(
            X_train_prep, X_val_prep, y_train, y_val, params
        )
        all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_path = os.path.join(SCRIPT_DIR, 'amazon_review_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"Results saved to: {output_path}")
    
    # Display summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(results_df.to_string())
    
    # Find best model
    if results_df['accuracy'].notna().any():
        best_idx = results_df['accuracy'].idxmax()
        best_result = results_df.loc[best_idx]
        best_params = eval(best_result['params'])
        
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Parameters: {best_result['params']}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        print(f"F1-Score: {best_result['f1']:.4f}")
        print(f"Training Time: {best_result['train_time']:.4f}s")
        
        # Create Kaggle submission with best model
        create_kaggle_submission(
            X_train_full, y_train_full, X_test_full, test_ids, best_params
        )
    
    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = run_amazon_review_experiments()
    
    print("\n" + "="*80)
    print("AMAZON REVIEW SVM EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print("  - High-dimensional data (10,000+ features)")
    print("  - Linear kernel should outperform RBF")
    print("  - Feature scaling is critical")
    print("  - 50-class classification is challenging")
    print("\nFiles created:")
    print("  - amazon_review_results.csv (experiment results)")
    print("  - kaggle_amazon_review.csv (Kaggle submission)")
    print("="*80)