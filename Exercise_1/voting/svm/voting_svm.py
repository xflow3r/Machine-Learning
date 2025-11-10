import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '../../Datasets')
TRAIN_PATH = os.path.join(DATASET_DIR, 'voting_learn.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'voting_test.csv')



def load_voting_dataset():

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Cannot find training data: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Cannot find test data: {TEST_PATH}")
    
    print(f"Loading training data from: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    
    print(f"Loading test data from: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    
    if 'class' in train_df.columns:
        y_train = train_df['class']
        X_train = train_df.drop(['class'], axis=1)
    else:
        y_train = train_df.iloc[:, -1]
        X_train = train_df.iloc[:, :-1]
    
    train_ids = X_train['ID'].values if 'ID' in X_train.columns else None
    test_ids = test_df['ID'].values if 'ID' in test_df.columns else None
    
    if 'ID' in X_train.columns:
        X_train = X_train.drop('ID', axis=1)
    if 'ID' in test_df.columns:
        X_test = test_df.drop('ID', axis=1)
    else:
        X_test = test_df.copy()
    
    if 'class' in X_test.columns:
        X_test = X_test.drop('class', axis=1)
    
    print(f"✓ Training data: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"✓ Test data: {len(X_test)} samples, {X_test.shape[1]} features")
    print(f"  Number of classes: {len(y_train.unique())}")
    print(f"  Classes: {sorted(y_train.unique())}")
    print(f"  Class distribution:")
    for cls, count in y_train.value_counts().items():
        print(f"    {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\n  Feature types:")
    numeric_cols = X_train.select_dtypes(include=[np.number]).shape[1]
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).shape[1]
    print(f"    Numeric: {numeric_cols}")
    print(f"    Categorical: {categorical_cols}")
    
    missing_train = X_train.isnull().sum().sum()
    missing_test = X_test.isnull().sum().sum()
    if missing_train > 0:
        print(f"  ⚠ Training data has {missing_train} missing values")
    if missing_test > 0:
        print(f"  ⚠ Test data has {missing_test} missing values")
    
    return X_train, y_train, X_test, test_ids, train_ids


def preprocess_voting_data(X_train, X_test, y_train):

    X_train = X_train.copy()
    X_test = X_test.copy()
    
    label_encoders = {}
    
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_train[col]):
            print(f"  Encoding categorical feature: {col}")
            
            le = LabelEncoder()
            
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            
            X_train[col] = X_train[col].replace('nan', 'MISSING')
            X_train[col] = X_train[col].replace('?', 'MISSING')
            X_test[col] = X_test[col].replace('nan', 'MISSING')
            X_test[col] = X_test[col].replace('?', 'MISSING')
            
            X_train[col] = le.fit_transform(X_train[col])
            label_encoders[col] = le
            
            X_test[col] = X_test[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    
    for col in X_train.columns:
        col_mean = X_train[col].mean()
        if pd.isna(col_mean):
            col_mean = 0
        X_train[col] = X_train[col].fillna(col_mean)
        
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(col_mean)
    
    print("  Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Preprocessing complete")
    print(f"  Training shape: {X_train_scaled.shape}")
    print(f"  Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, label_encoders, scaler



def get_svm_param_grid():

    params = [
        {'kernel': 'linear', 'C': 0.1},
        {'kernel': 'linear', 'C': 1.0},
        {'kernel': 'linear', 'C': 10.0},
        {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
    ]
    return params


def train_and_evaluate_svm(X_train, X_val, y_train, y_val, params):

    print(f"\n  Testing SVM with {params}...")
    
    result = {
        'params': str(params),
        'kernel': params['kernel'],
        'C': params['C'],
    }
    
    try:
        svm = SVC(**params, random_state=42)
        
        start_time = time.time()
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = svm.predict(X_val)
        
        result['accuracy'] = accuracy_score(y_val, y_pred)
        result['train_time'] = train_time
        
        try:
            result['precision'] = precision_score(y_val, y_pred, average='binary', pos_label='democrat', zero_division=0)
            result['recall'] = recall_score(y_val, y_pred, average='binary', pos_label='democrat', zero_division=0)
            result['f1'] = f1_score(y_val, y_pred, average='binary', pos_label='democrat', zero_division=0)
            
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


def create_kaggle_submission(X_train_full, y_train_full, X_test, test_ids, best_params):

    X_train_prep, X_test_prep, label_encoders, scaler = preprocess_voting_data(
        X_train_full, X_test, y_train_full
    )
    
    print("  Training final SVM model...")
    svm = SVC(**best_params, random_state=42)
    start_time = time.time()
    svm.fit(X_train_prep, y_train_full)
    train_time = time.time() - start_time
    print(f"  ✓ Training complete in {train_time:.4f}s")
    
    print("  Making predictions on test set...")
    y_pred = svm.predict(X_test_prep)
    print(f"  ✓ Generated {len(y_pred)} predictions")
    
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f"  Prediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"    {cls}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'class': y_pred
    })
    
    output_path = os.path.join(SCRIPT_DIR, 'kaggle_voting.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"\n✓ Kaggle submission saved: {output_path}")
    print(f"  Format: ID, class")
    print(f"  Rows: {len(submission)}")
    print(f"  Ready to upload to Kaggle!")
    
    return output_path

def run_voting_experiments():

    X_train_full, y_train_full, X_test_full, test_ids, train_ids = load_voting_dataset()
    
    print("\n" + "="*80)
    print("CREATING TRAIN/VALIDATION SPLIT")
    print("="*80)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
    except:

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Validation class distribution:")
    for cls, count in y_val.value_counts().items():
        print(f"    {cls}: {count} ({count/len(y_val)*100:.1f}%)")
    
    X_train_prep, X_val_prep, label_encoders, scaler = preprocess_voting_data(
        X_train, X_val, y_train
    )
    
    params_grid = get_svm_param_grid()
    
    
    all_results = []
    
    for params in params_grid:
        result = train_and_evaluate_svm(
            X_train_prep, X_val_prep, y_train, y_val, params
        )
        all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    output_path = os.path.join(SCRIPT_DIR, 'voting_results.csv')
    results_df.to_csv(output_path, index=False)
    
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
        
        create_kaggle_submission(
            X_train_full, y_train_full, X_test_full, test_ids, best_params
        )
    
    return results_df

if __name__ == "__main__":
    results = run_voting_experiments()
    

    print("  - voting_results.csv (experiment results)")
    print("  - kaggle_voting.csv (Kaggle submission)")
    print("="*80)
