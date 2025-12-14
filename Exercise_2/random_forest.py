#!/usr/bin/env python3

import numpy as np
from joblib import Parallel, delayed
from regression_tree import RegressionTree


class RandomForest:
    """
    Random Forest Regressor using bagging and feature randomness
    """
    
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None, n_jobs=-1):
        """
        Initialize Random Forest
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features # number of features to consider at each split
        self.n_jobs = n_jobs # number of parallel jobs (-1 = use all cores)
        self.trees = []

    
    def fit(self, X, y):
        """
        Build forest of trees from training data (parallelized)
        """
        # Train trees in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_tree)(X, y, seed) 
            for seed in range(self.n_trees)
        )
    
    
    def _train_tree(self, X, y, seed):
        """
        Train a single tree (helper for parallelization)
        """
        # Each _train_tree call is a separate process, so need to call seed() for each tree
        np.random.seed(seed)
        X_sample, y_sample = self._bootstrap_sample(X, y)
        tree = RegressionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            in_random_forest=True,
            max_features=self.max_features
        )
        tree.fit(X_sample, y_sample)
        return tree

    
    def predict(self, X):
        """
        Predict target values by averaging predictions from all trees
        """
        # Aggregate predictions from all trees, and return average
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
    
    
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample (random sampling with replacement)
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    
    
    def get_params(self, deep=True):
        """
        Get parameters (sklearn compatibility)
        """
        return {
            'n_trees': self.n_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'n_jobs': self.n_jobs
        }
    
    
    def set_params(self, **params):
        """
        Set parameters (sklearn compatibility)
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from joblib import dump
    
    parser = argparse.ArgumentParser(description='Random Forest Regressor')
    parser.add_argument('-n', '--n_trees', type=int, default=100, help='Number of trees in the forest')
    parser.add_argument('-d', '--max_depth', type=int, default=10, help='Maximum depth of each tree')
    parser.add_argument('-s', '--min_samples_split', type=int, default=2, help='Minimum samples required to split a node')
    parser.add_argument('-l', '--min_samples_leaf', type=int, default=1, help='Minimum samples required at a leaf node')
    parser.add_argument('-f', '--max_features', type=int, default=None, help='Number of features to consider at each split (default: sqrt(n_features))')
    parser.add_argument('-j', '--n_jobs', type=int, default=-1, help='Number of parallel jobs (-1 uses all cores)')
    
    args = parser.parse_args()
    
    datasets = {
        'Bikes': ('datasets/preprocessed_bikes.csv', 'Rented Bike Count'),
        'Cars': ('datasets/preprocessed_cars.csv', 'price_usd'),
        'Houses': ('datasets/preprocessed_houses.csv', 'SalePrice')
    }
    
    print(f"Random Forest Parameters: n_trees={args.n_trees}, max_depth={args.max_depth}, "
          f"min_samples_split={args.min_samples_split}, min_samples_leaf={args.min_samples_leaf}, "
          f"max_features={args.max_features}")
    print()
    
    for name, (path, target) in datasets.items():
        print(f"Testing Random Forest on {name} Dataset")

        df = pd.read_csv(path)
        
        X = df.drop(target, axis=1).values
        y = df[target].values
        
        forest = RandomForest(
            n_trees=args.n_trees,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=args.n_jobs
        )
        
        mse_scores = -cross_val_score(forest, X, y, cv=5, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(forest, X, y, cv=5, scoring='r2')
        
        print(f"MSE: {mse_scores.mean():.2f} (+/- {mse_scores.std():.2f})")
        print(f"R2: {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
        
        # Check if model already exists
        params_str = f"n{args.n_trees}_d{args.max_depth}_s{args.min_samples_split}_l{args.min_samples_leaf}_f{args.max_features}"
        model_filename = f"{name}_trained_model.{params_str}.joblib"
        
        if os.path.exists(model_filename):
            print(f"Model {model_filename} already exists, skipping full dataset training + saving.")
        else:
            # Train on full dataset
            print(f"Training on full {name} dataset...")
            forest.fit(X, y)
            
            # Save model
            dump(forest, model_filename)
            print(f"Model saved to {model_filename}")
        print()
