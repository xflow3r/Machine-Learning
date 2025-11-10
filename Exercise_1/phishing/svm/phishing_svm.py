"""
SVM Classification for Phishing Dataset
Exercise 1 - Machine Learning
Dataset: Phishing (ARFF format)
- 1,353 samples, 9 categorical features, 3 classes
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from scipy.io import arff
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, '../../Datasets/phishing_data.arff')

print("="*80)
print("PHISHING DATASET - SVM CLASSIFICATION")
print("="*80)
print(f"Script directory: {SCRIPT_DIR}")
print(f"Dataset path: {DATASET_PATH}")


# ============================================================================
# ARFF FILE LOADING FUNCTIONS
# ============================================================================

def load_arff_file(filepath):
    """
    Load ARFF file and convert to pandas DataFrame
    Handles string attributes properly
    """
    try:
        # Try loading with scipy
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        
        # Convert byte strings to regular strings
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].str.decode('utf-8')
                except AttributeError:
                    # Already strings or other type
                    pass
        
        return df
    
    except Exception as e:
        if "String attributes not supported" in str(e):
            print(f"   ⚠ String attributes detected, using alternative loader...")
            return load_arff_with_pandas(filepath)
        else:
            raise e


def load_arff_with_pandas(filepath):
    """
    Alternative ARFF loader using pandas for files with string attributes
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find @DATA line
    data_start = 0
    attributes = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith('@attribute'):
            # Parse attribute name
            parts = line.split()
            attr_name = parts[1].strip("'\"")
            attributes.append(attr_name)
        elif line.lower().startswith('@data'):
            data_start = i + 1
            break
    
    # Read data portion
    data_lines = [line.strip() for line in lines[data_start:] 
                  if line.strip() and not line.strip().startswith('%')]
    
    # Create DataFrame
    data_rows = []
    for line in data_lines:
        # Handle both comma-separated and quoted values
        values = []
        current_val = ""
        in_quotes = False
        
        for char in line:
            if char == '"' or char == "'":
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                values.append(current_val.strip())
                current_val = ""
            else:
                current_val += char
        values.append(current_val.strip())  # Add last value
        
        data_rows.append(values)
    
    df = pd.DataFrame(data_rows, columns=attributes)
    
    # Try to convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            # Keep as string/categorical
            pass
    
    return df


# ============================================================================
# DATA LOADING
# ============================================================================

def load_phishing_dataset():
    """
    Load the Phishing dataset
    Returns: X (features), y (target)
    """
    print("\n" + "="*80)
    print("LOADING PHISHING DATASET")
    print("="*80)
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Cannot find dataset: {DATASET_PATH}")
    
    print(f"Loading from: {DATASET_PATH}")
    df = load_arff_file(DATASET_PATH)
    
    # Last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    print(f"✓ Loaded: {len(X)} samples, {X.shape[1]} features")
    print(f"  Features: {list(X.columns)}")
    print(f"  Classes: {sorted(y.unique())}")
    print(f"  Class distribution:")
    for cls, count in y.value_counts().items():
        print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    return X, y


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_phishing_data(X_train, X_test, y_train):
    """
    Preprocess Phishing dataset:
    - Encode all categorical features
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
        # Check if column is object/string type
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
        X_test[col] = X_test[col].fillna(col_mean)
    
    # Scale features (CRITICAL for SVM!)
    print("  Scaling features...")
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

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, params):
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
        
        # Predict
        y_pred = svm.predict(X_test)
        
        # Metrics
        result['accuracy'] = accuracy_score(y_test, y_pred)
        result['train_time'] = train_time
        
        # Additional metrics for multi-class
        try:
            result['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            result['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            result['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
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
# MAIN EXPERIMENT
# ============================================================================

def run_phishing_experiments():
    """
    Main function to run all SVM experiments on Phishing dataset
    """
    print("\n" + "="*80)
    print("STARTING SVM EXPERIMENTS")
    print("="*80)
    
    # Load dataset
    X, y = load_phishing_dataset()
    
    # Train/validation split (80/20)
    print("\n" + "="*80)
    print("CREATING TRAIN/VALIDATION SPLIT")
    print("="*80)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    
    # Preprocess
    X_train_prep, X_val_prep, label_encoders, scaler = preprocess_phishing_data(
        X_train, X_val, y_train
    )
    
    # Get parameter grid
    params_grid = get_svm_param_grid()
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    all_results = []
    
    for params in params_grid:
        result = train_and_evaluate_svm(
            X_train_prep, X_val_prep, y_train, y_val, params
        )
        all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_path = os.path.join(SCRIPT_DIR, 'phishing_results.csv')
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
        
        print("\n" + "="*80)
        print("BEST MODEL")
        print("="*80)
        print(f"Parameters: {best_result['params']}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        print(f"F1-Score: {best_result['f1']:.4f}")
        print(f"Training Time: {best_result['train_time']:.4f}s")
    
    return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = run_phishing_experiments()
    
    print("\n" + "="*80)
    print("PHISHING SVM EXPERIMENTS COMPLETE!")
    print("="*80)