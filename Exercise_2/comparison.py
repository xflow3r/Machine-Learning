#!/usr/bin/env python3
"""
Comparison script for regression algorithms.
Compares custom implementations with sklearn implementations.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from regression_tree import RegressionTree
from random_forest import RandomForest


def evaluate_model(model, X, y, model_name, cv=5):
    """
    Evaluate a model using cross-validation with timing metrics.
    
    Returns:
        dict: Contains mean and std for MSE, R2, training time, and prediction time
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    mse_scores = []
    r2_scores = []
    train_times = []
    predict_times = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        train_times.append(train_time)
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_val)
        predict_time = time.time() - start_time
        predict_times.append(predict_time)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        mse_scores.append(mse)
        r2_scores.append(r2)
    
    return {
        'model': model_name,
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'train_time_mean': np.mean(train_times),
        'train_time_std': np.std(train_times),
        'predict_time_mean': np.mean(predict_times),
        'predict_time_std': np.std(predict_times)
    }


def main():
    # Load datasets
    datasets = {
        'Bikes': ('datasets/preprocessed_bikes.csv', 'Rented Bike Count'),
        'Cars': ('datasets/preprocessed_cars.csv', 'price_usd'),
        'Houses': ('datasets/preprocessed_houses.csv', 'SalePrice')
    }
    
    # Define models to compare
    models = [
        # Custom implementations
        (RegressionTree(max_depth=10, min_samples_split=10, min_samples_leaf=5), 
         "Custom RegressionTree"),
        
        (RandomForest(n_trees=10, max_depth=5, min_samples_split=10, min_samples_leaf=5, n_jobs=-1), 
         "Custom RandomForest (n=10, d=5)"),
        
        (RandomForest(n_trees=50, max_depth=10, min_samples_split=10, min_samples_leaf=5, n_jobs=-1), 
         "Custom RandomForest (n=50, d=10)"),
        
        (RandomForest(n_trees=100, max_depth=15, min_samples_split=10, min_samples_leaf=5, n_jobs=-1), 
         "Custom RandomForest (n=100, d=15)"),
        
        # Sklearn implementations
        (DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42), 
         "sklearn DecisionTreeRegressor"),
        
        (RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=5, 
                               n_jobs=-1, random_state=42), 
         "sklearn RandomForestRegressor"),
        
        (GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42), 
         "sklearn GradientBoostingRegressor")
    ]
    
    # Store all results
    all_results = []
    
    # Run comparisons for each dataset
    for dataset_name, (path, target) in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Load data
        df = pd.read_csv(path)
        X = df.drop(target, axis=1).values
        y = df[target].values
        
        print(f"Shape: {X.shape[0]} samples, {X.shape[1]} features")
        print()
        
        # Evaluate each model
        for model, model_name in models:
            print(f"Evaluating: {model_name}...")
            
            result = evaluate_model(model, X, y, model_name, cv=5)
            result['dataset'] = dataset_name
            all_results.append(result)
            
            print(f"  MSE: {result['mse_mean']:.2f} (+/- {result['mse_std']:.2f})")
            print(f"  R²: {result['r2_mean']:.4f} (+/- {result['r2_std']:.4f})")
            print(f"  Training time: {result['train_time_mean']:.4f}s (+/- {result['train_time_std']:.4f}s)")
            print(f"  Prediction time: {result['predict_time_mean']:.6f}s (+/- {result['predict_time_std']:.6f}s)")
            print()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    columns_order = ['dataset', 'model', 'r2_mean', 'r2_std', 'mse_mean', 'mse_std', 
                     'train_time_mean', 'train_time_std', 'predict_time_mean', 'predict_time_std']
    results_df = results_df[columns_order]
    
    # Save to CSV
    output_file = 'comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Average Performance Across All Datasets")
    print("="*80)
    
    summary = results_df.groupby('model').agg({
        'r2_mean': 'mean',
        'mse_mean': 'mean',
        'train_time_mean': 'mean',
        'predict_time_mean': 'mean'
    }).round(4)
    
    summary.columns = ['Avg R²', 'Avg MSE', 'Avg Train Time (s)', 'Avg Predict Time (s)']
    summary = summary.sort_values('Avg R²', ascending=False)
    
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
