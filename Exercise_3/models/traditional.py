"""
PERSON A: Traditional machine learning methods implementation
Implements HOG feature extraction with SVM and Logistic Regression classifiers
"""

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from skimage.feature import hog
from tqdm import tqdm


def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=False):
    """
    Extract HOG features from images

    Args:
        images: numpy array of shape (N, 28, 28)
        orientations: number of orientation bins
        pixels_per_cell: size of a cell
        cells_per_block: number of cells in each block
        visualize: whether to return HOG image

    Returns:
        numpy array of HOG features
    """
    features = []

    print(f"Extracting HOG features from {len(images)} images...")
    for img in tqdm(images, desc="HOG extraction"):
        # Extract HOG features
        fd = hog(img, orientations=orientations,
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block,
                 visualize=visualize, channel_axis=None)
        features.append(fd)

    return np.array(features)


def run_traditional(model_name, X_train, y_train, X_test, y_test):
    """
    Run traditional ML methods (HOG + SVM/LogReg)

    Args:
        model_name: str, one of {"hog_svm", "hog_logreg"}
        X_train: numpy array of shape (N, 28, 28)
        y_train: numpy array of shape (N,)
        X_test: numpy array of shape (M, 28, 28)
        y_test: numpy array of shape (M,)

    Returns:
        dict containing results
    """
    results = {}

    # Step 1: Extract HOG features
    print("\n=== Feature Extraction ===")
    feature_start = time.time()

    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    feature_time = time.time() - feature_start
    print(f"Feature extraction completed in {feature_time:.2f}s")
    print(f"Feature shape: {X_train_hog.shape}")

    # Step 2: Train classifier
    print("\n=== Training Classifier ===")
    train_start = time.time()

    if model_name == "hog_svm":
        print("Training SVM with RBF kernel...")
        classifier = SVC(kernel='rbf', C=10.0, gamma='scale', verbose=False)
        classifier.fit(X_train_hog, y_train)

    elif model_name == "hog_logreg":
        print("Training Logistic Regression...")
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs',
                                        multi_class='multinomial', verbose=0)
        classifier.fit(X_train_hog, y_train)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f}s")

    # Step 3: Make predictions
    print("\n=== Evaluation ===")
    test_start = time.time()

    print("Making predictions on test set...")
    y_pred = classifier.predict(X_test_hog)

    test_time = time.time() - test_start
    print(f"Prediction completed in {test_time:.2f}s")

    # Compile results
    results = {
        'accuracy': None,  # Will be computed in main.py
        'f1_macro': None,  # Will be computed in main.py
        'train_time': train_time,
        'test_time': test_time,
        'feature_time': feature_time,
        'y_true': y_test,
        'y_pred': y_pred,
        'model_name': model_name
    }

    return results
