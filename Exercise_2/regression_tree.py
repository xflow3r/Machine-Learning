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
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        """
        Constructor
        """
        self.max_depth = max_depth # max tree depth (too deep = overfitting, too shallow = underfitting)
        self.min_samples_split = min_samples_split # samples needed to split (prevents splitting of too small groups)
        self.min_samples_leaf = min_samples_leaf # samples needed in a leaf node (prevents leafes with to few samples which makes predictions more stable)
        
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
        best_mse = float('inf')  # Start with infinity (worst possible)
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature_index in range(n_features):
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
                
                mse = self._calculate_mse(y_left, y_right)
                
                if mse < best_mse: # update mse if it improved
                    best_mse = mse
                    best_feature = feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    
    def _calculate_mse(self, y_left, y_right):
        """
        Calculate weighted Mean Squared Error for a split
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
