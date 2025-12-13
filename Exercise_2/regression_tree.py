#!/usr/bin/env python3
import numpy as np

class TreeNode:
    """
    TreeNode in the tree. TreeNode OR LeafNode(end of tree)
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        
        self.feature_index = feature_index  # Which feature is currently worked on
        self.threshold = threshold          # Threshold to decide the left/right split
        self.left = left                    # Left TreeNode
        self.right = right                  # Right TreeNode
        
        # For leaf nodes
        self.value = value                  # The prediction (mean of y values)

    
    def is_leaf_node(self):
        """
        Check if this node is a leaf
        """
        return self.value is not None # returns true/false depending if we reached end of tree


class RegressionTree:
    """
    Regression Tree for predicting continuous values
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, in_random_forest=False, max_features=None):
        """
        Constructor
        """
        self.max_depth = max_depth # max tree depth (too deep = overfitting, too shallow = underfitting)
        self.min_samples_split = min_samples_split # samples needed to split (prevents splitting of too small groups)
        self.min_samples_leaf = min_samples_leaf # samples needed in a leaf node (prevents leafes with to few samples which makes predictions more stable)
        self.in_random_forest = in_random_forest # whether to use random feature selection at each node
        self.max_features = max_features # number of features to consider at each split (only used if in_random_forest=True)
        
        self.root = None

    
    def fit(self, X, y):
        """
        Build the tree from training data
        """
        X = np.array(X)
        y = np.array(y)

        self.root = self._build_tree(X, y, depth=0)

    
    def predict(self, X):
        """
        Predict target values for samples in X
        """
        X = np.array(X)
        predictions = [self._traverse_tree(x, self.root) for x in X]

        return np.array(predictions)

    
    def _build_tree(self, X, y, depth):
        """
        build the tree
        """
        n_samples, n_features = X.shape
    
        if depth >= self.max_depth:
            return self._create_leaf(y)
        
        if n_samples < self.min_samples_split:
            return self._create_leaf(y)
        
        if len(np.unique(y)) == 1:
            return self._create_leaf(y)
        
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # no split found -> create leaf
        if best_feature is None:
            return self._create_leaf(y)
        
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature, best_threshold)
        
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
    
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on
        """
        best_variance = float('inf')  # Start with infinity (worst possible)
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        # Select feature subset if in random forest mode
        if self.in_random_forest:
            # Determine number of features to consider
            if self.max_features is None:
                # Default: sqrt(n_features) for random forests
                n_features_to_consider = max(1, int(np.sqrt(n_features)))
            else:
                n_features_to_consider = min(self.max_features, n_features)
            
            # Randomly select feature indices
            feature_indices = np.random.choice(n_features, size=n_features_to_consider, replace=False)
        else:
            # Use all features
            feature_indices = range(n_features)
        
        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            
            possible_thresholds = np.unique(feature_values) # unique values for possible thresholds
            
            for threshold in possible_thresholds:

                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0: # ignore impossible split (one side empty)
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                variance = self._calculate_variance(y_left, y_right)
                
                if variance < best_variance: # update best variance, if it improved
                    best_variance = variance
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    
    def _calculate_variance(self, y_left, y_right):
        """
        Calculate weighted variance for a split
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        var_left = np.var(y_left)
        var_right = np.var(y_right)
        
        weighted_mse = (n_left / n_total) * var_left + (n_right / n_total) * var_right
        
        return weighted_mse

    
    
    def _split_data(self, X, y, feature_index, threshold):
        """
        Split data based on feature and threshold
        """
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        
        # Split the data
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        return X_left, y_left, X_right, y_right

    def get_params(self, deep=True): # included because of sklearn
        """
        Get parameters for this estimator (required for sklearn compatibility)
        """
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'in_random_forest': self.in_random_forest,
            'max_features': self.max_features
        }

    def set_params(self, **params): # included because of sklearn
        """
        Set parameters for this estimator (required for sklearn compatibility)
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def _create_leaf(self, y):
        """
        Create a leaf node with prediction value
        """
        leaf_value = np.mean(y)
    
        return TreeNode(value=leaf_value)

    
    def _traverse_tree(self, x, node):
        """
        Traverse tree for a single sample to get prediction
        """

        # found the node
        if node.is_leaf_node(): 
            return node.value
        
        # search tree
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left) # left
        else:
            return self._traverse_tree(x, node.right) # right

    
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, mean_squared_error, r2_score

    datasets = {
        'Bikes': ('datasets/preprocessed_bikes.csv', 'Rented Bike Count'),
        'Cars': ('datasets/preprocessed_cars.csv', 'price_usd'),
        'Houses': ('datasets/preprocessed_houses.csv', 'SalePrice')
    }
    
    for name, (path, target) in datasets.items():
        print(f"Testing on {name} Dataset")
        
        df = pd.read_csv(path)
        
        X = df.drop(target, axis=1).values
        y = df[target].values
        
        tree = RegressionTree(max_depth=5, min_samples_split=10, min_samples_leaf=5)
        
        # TODO: may need to update RegressionTree to inherit from BaseEstimator for sklearn compatibility,
        # else add # type: ignore
        mse_scores = -cross_val_score(tree, X, y, cv=5, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(tree, X, y, cv=5, scoring='r2')
        
        print(f"MSE: {mse_scores.mean():.2f} (+/- {mse_scores.std():.2f})")
        print(f"R2: {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")