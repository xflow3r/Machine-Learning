"""
SVM Classification for Voting Dataset
Exercise 1 - Machine Learning
Dataset: Voting Records (CSV format)
- 218 training samples, 17 features (mostly categorical), 2 classes
- Binary classification: Democrat vs Republican
- Kaggle competition dataset
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '../../Datasets')
TRAIN_PATH = os.path.join(DATASET_DIR, 'voting_learn.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'voting_test.csv')

print("="*80)
print("VOTING DATASET - SVM CLASSIFICATION")
print("="*80)
print(f"Script directory: {SCRIPT_DIR}")
print(f"Training data: {TRAIN_PATH}")
print(f"Test data: {TEST_PATH}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_voting_dataset():
    """
    Load the Voting dataset (CSV format)
    Returns: X_train, y_train, X_test, test_ids, train_ids
    """
    print("\n" + "="*80)
    print("LOADING VOTING DATASET")
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
    # Target is 'class' column
    if 'class' in train_df.columns:
        y_train = train_df['class']
        X_train = train_df.drop(['class'], axis=1)
    else:
        # Fallback: last column is target
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
    
    # Remove 'class' from test set if it exists
    if 'class' in X_test.columns:
        X_test = X_test.drop('class', axis=1)
    
    print(f"✓ Training data: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"✓ Test data: {len(X_test)} samples, {X_test.shape[1]} features")
    print(f"  Number of classes: {len(y_train.unique())}")
    print(f"  Classes: {sorted(y_train.unique())}")
    print(f"  Class distribution:")
    for cls, count in y_train.value_counts().items():
        print(f"    {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Show feature types
    print(f"\n  Feature types:")
    numeric_cols = X_train.select_dtypes(include=[np.number]).shape[1]
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).shape[1]
    print(f"    Numeric: {numeric_cols}")
    print(f"    Categorical: {categorical_cols}")
    
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

def preprocess_voting_data(X_train, X_test, y_train):
    """
    Preprocess Voting dataset:
    - Encode categorical features (most are categorical)
    - Handle missing values
    - Scale features
    
    Returns: X_train_scaled, X_test_scaled, label_encoders, scaler
    """
    print("\n" + "="*80)
    print("PREPROCESSING")
    print("="*80)
    
    # Make copies
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Handle categorical features with Label Encoding
    label_encoders = {}
    
    for col in X_train.columns:
        # Check if column is object/string type or has non-numeric values
        if X_train[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_train[col]):
            print(f"  Encoding categorical feature: {col}")
            
            # Create label encoder for this column
            le = LabelEncoder()
            
            # Convert to string first to handle mixed types
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            
            # Handle missing values as a category
            X_train[col] = X_train[col].replace('nan', 'MISSING')
            X_train[col] = X_train[col].replace('?', 'MISSING')
            X_test[col] = X_test[col].replace('nan', 'MISSING')
            X_test[col] = X_test[col].replace('?', 'MISSING')
            
            # Fit and transform training data
            X_train[col] = le.fit_transform(X_train[col])
            label_encoders[col] = le
            
            # Transform test data, handle unseen categories
            X_test[col] = X_test[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    
    # Convert everything to numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    
    # Fill any remaining NaN values with column mean
    for col in X_train.columns:
        col_mean = X_train[col].mean()
        if pd.isna(col_mean):
            col_mean = 0
        X_train[col] = X_train[col].fillna(col_mean)
        
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(col_mean)
    
    # Scale features (CRITICAL for SVM!)
    print("  Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Preprocessing complete")
    print(f"  Training shape: {X_train_scaled.shape}")
    print(f"  Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, label_encoders, scaler


# ============================================================================
# SVM PARAMETER GRID
# ============================================================================

def get_svm_param_grid():
    """
    Define SVM parameter combinations to test
    For small, low-dimensional data, both Linear and RBF can work well
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
        
        # Binary classification metrics
        try:
            result['precision'] = precision_score(y_val, y_pred, average='binary', pos_label='democrat', zero_division=0)
            result['recall'] = recall_score(y_val, y_pred, average='binary', pos_label='democrat', zero_division=0)
            result['f1'] = f1_score(y_val, y_pred, average='binary', pos_label='democrat', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            result['confusion_matrix'] = str(cm.tolist())
        except:
            result['precision'] = np.nan
            result['recall'] = np.nan
            result['f1'] = np.nan
            result['confusion_matrix'] = 'N/A'
        
        print(f"    ✓ Accuracy: {result['accuracy']:.4f}, Time: {train_time:.4f}s")
        
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        result['error'] = str(e)
        result['accuracy'] = np.nan
        result['train_time'] = np.nan
        result['precision'] = np.nan
        result['recall'] = np.nan
        result['f1'] = np.nan
        result['confusion_matrix'] = 'N/A'
    
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
    X_train_prep, X_test_prep, label_encoders, scaler = preprocess_voting_data(
        X_train_full, X_test, y_train_full
    )
    
    # Train final model
    print("  Training final SVM model...")
    svm = SVC(**best_params, random_state=42)
    start_time = time.time()
    svm.fit(X_train_prep, y_train_full)
    train_time = time.time() - start_time
    print(f"  ✓ Training complete in {train_time:.4f}s")
    
    # Predict on test set
    print("  Making predictions on test set...")
    y_pred = svm.predict(X_test_prep)
    print(f"  ✓ Generated {len(y_pred)} predictions")
    
    # Show prediction distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"  Prediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"    {cls}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'ID': test_ids,
        'class': y_pred
    })
    
    # Save submission
    output_path = os.path.join(SCRIPT_DIR, 'kaggle_voting.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Kaggle submission saved: {output_path}")
    print(f"  Format: ID, class")
    print(f"  Rows: {len(submission)}")
    print(f"  Ready to upload to Kaggle!")
    
    return output_path


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_voting_experiments():
    """
    Main function to run all SVM experiments on Voting dataset
    """
    print("\n" + "="*80)
    print("STARTING SVM EXPERIMENTS")
    print("="*80)
    
    # Load dataset
    X_train_full, y_train_full, X_test_full, test_ids, train_ids = load_voting_dataset()
    
    # Create train/validation split for experiments (80/20)
    print("\n" + "="*80)
    print("CREATING TRAIN/VALIDATION SPLIT")
    print("="*80)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
    except:
        # If stratification fails (unlikely with binary classification)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Validation class distribution:")
    for cls, count in y_val.value_counts().items():
        print(f"    {cls}: {count} ({count/len(y_val)*100:.1f}%)")
    
    # Preprocess
    X_train_prep, X_val_prep, label_encoders, scaler = preprocess_voting_data(
        X_train, X_val, y_train
    )
    
    # Get parameter grid
    params_grid = get_svm_param_grid()
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    print("NOTE: Small dataset (218 samples), binary classification")
    print("      Both Linear and RBF kernels should perform well")
    
    all_results = []
    
    for params in params_grid:
        result = train_and_evaluate_svm(
            X_train_prep, X_val_prep, y_train, y_val, params
        )
        all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_path = os.path.join(SCRIPT_DIR, 'voting_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"Results saved to: {output_path}")
    
    # Display summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(results_df[['kernel', 'C', 'accuracy', 'precision', 'recall', 'f1', 'train_time']].to_string())
    
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
        print(f"Training Time: {best_result['train_time']:.6f}s")
        
        # Create Kaggle submission with best model
        create_kaggle_submission(
            X_train_full, y_train_full, X_test_full, test_ids, best_params
        )
    
    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = run_voting_experiments()
    
    print("\n" + "="*80)
    print("VOTING SVM EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print("  - Small dataset (218 samples)")
    print("  - Binary classification (Democrat vs Republican)")
    print("  - 16 categorical features")
    print("  - Very fast training (<0.01s)")
    print("  - High accuracy achievable (>95%)")
    print("\nFiles created:")
    print("  - voting_results.csv (experiment results)")
    print("  - kaggle_voting.csv (Kaggle submission)")
    print("="*80)