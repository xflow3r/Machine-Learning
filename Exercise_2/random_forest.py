import numpy as np
from regression_tree import RegressionTree


class RandomForest:
    """
    Random Forest Regressor using bagging and feature randomness
    """
    
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None, bootstrap=True):
        """
        Initialize Random Forest
        """
        pass
    
    
    def fit(self, X, y):
        """
        Build forest of trees from training data
        """
        pass
    
    
    def predict(self, X):
        """
        Predict target values by averaging predictions from all trees
        """
        pass
    
    
    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample (random sampling with replacement)
        """
        pass
    
    
    def _get_random_features(self, n_features):
        """
        Select random subset of features for a tree
        """
        pass
    
    
    def get_params(self, deep=True):
        """
        Get parameters (sklearn compatibility)
        """
        pass
    
    
    def set_params(self, **params):
        """
        Set parameters (sklearn compatibility)
        """
        pass


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    
    datasets = {
        'Bikes': ('datasets/preprocessed_bikes.csv', 'Rented Bike Count'),
        'Cars': ('datasets/preprocessed_cars.csv', 'price_usd'),
        'Houses': ('datasets/preprocessed_houses.csv', 'SalePrice')
    }
    
    for name, (path, target) in datasets.items():
        print(f"Testing Random Forest on {name} Dataset")
        
        df = pd.read_csv(path)
        
        X = df.drop(target, axis=1).values
        y = df[target].values
        
        forest = RandomForest(n_trees=10, max_depth=5, min_samples_split=10, min_samples_leaf=5)
        
        mse_scores = -cross_val_score(forest, X, y, cv=5, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(forest, X, y, cv=5, scoring='r2')
        
        print(f"MSE: {mse_scores.mean():.2f} (+/- {mse_scores.std():.2f})")
        print(f"R2: {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
        print()
