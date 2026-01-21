"""
PERSON A: Implement traditional methods here

Required function signature:
    run_traditional(model_name, X_train, y_train, X_test, y_test) -> dict

Args:
    model_name: str, one of {"hog_svm", "hog_logreg"}
    X_train: numpy array of shape (N, 28, 28)
    y_train: numpy array of shape (N,)
    X_test: numpy array of shape (M, 28, 28)
    y_test: numpy array of shape (M,)

Returns:
    dict containing:
        - 'accuracy': float
        - 'f1_macro': float
        - 'train_time': float (seconds)
        - 'test_time': float (seconds)
        - 'feature_time': float (seconds, HOG extraction time)
        - 'y_true': numpy array (ground truth labels)
        - 'y_pred': numpy array (predicted labels)
        - 'model_name': str
"""


def run_traditional(model_name, X_train, y_train, X_test, y_test):
    """
    Run traditional ML methods (HOG + SVM/LogReg)

    TODO for Person A:
    1. Implement HOG feature extraction
    2. Implement SVM classifier (model_name == "hog_svm")
    3. Implement Logistic Regression (model_name == "hog_logreg")
    4. Return results in the required format
    """
    raise NotImplementedError("Person A needs to implement this function")