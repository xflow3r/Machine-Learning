"""
Traditional machine learning methods implementation
Implements:
- Color histogram features (simple baseline)
- HOG feature extraction (powerful keypoint-based)
- SVM and Logistic Regression classifiers
"""

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from skimage.feature import hog
from tqdm import tqdm


def extract_color_histogram(images, bins=32):
    """
    Extract color histogram features from images
    Simple baseline feature representation

    Args:
        images: numpy array of shape (N, H, W) for grayscale or (N, H, W, 3) for RGB
        bins: number of bins per channel

    Returns:
        numpy array of histogram features
    """
    features = []
    is_grayscale = (len(images.shape) == 3)

    print(f"Extracting color histogram features from {len(images)} images...")
    for img in tqdm(images, desc="Histogram extraction"):
        if is_grayscale:
            # Grayscale: single histogram
            hist, _ = np.histogram(img.flatten(), bins=bins, range=(0, 1))
            hist = hist.astype(float) / hist.sum()  # Normalize
            features.append(hist)
        else:
            # RGB: concatenate histograms from all 3 channels
            hist_r, _ = np.histogram(img[:, :, 0].flatten(), bins=bins, range=(0, 1))
            hist_g, _ = np.histogram(img[:, :, 1].flatten(), bins=bins, range=(0, 1))
            hist_b, _ = np.histogram(img[:, :, 2].flatten(), bins=bins, range=(0, 1))

            # Normalize each histogram
            hist_r = hist_r.astype(float) / hist_r.sum()
            hist_g = hist_g.astype(float) / hist_g.sum()
            hist_b = hist_b.astype(float) / hist_b.sum()

            # Concatenate all channels
            hist = np.concatenate([hist_r, hist_g, hist_b])
            features.append(hist)

    return np.array(features)


def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=False):
    """
    Extract HOG features from images
    Powerful keypoint-based feature representation

    Args:
        images: numpy array of shape (N, H, W) for grayscale or (N, H, W, 3) for RGB
        orientations: number of orientation bins
        pixels_per_cell: size of a cell
        cells_per_block: number of cells in each block
        visualize: whether to return HOG image

    Returns:
        numpy array of HOG features
    """
    features = []
    is_grayscale = (len(images.shape) == 3)

    print(f"Extracting HOG features from {len(images)} images...")
    for img in tqdm(images, desc="HOG extraction"):
        if is_grayscale:
            # Grayscale image
            fd = hog(img, orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    visualize=visualize, channel_axis=None)
        else:
            # RGB image - extract HOG from each channel and concatenate
            fd = hog(img, orientations=orientations,
                    pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block,
                    visualize=visualize, channel_axis=-1)
        features.append(fd)

    return np.array(features)


def run_traditional(model_name, X_train, y_train, X_test, y_test):
    """
    Run traditional ML methods

    Note: These methods use CPU only (scikit-learn doesn't support GPU).
    For GPU acceleration, use the CNN models.

    Models:
    - hist_svm: Color Histogram + SVM (simple baseline)
    - hist_logreg: Color Histogram + Logistic Regression (simple baseline)
    - hog_svm: HOG + SVM (powerful approach)
    - hog_logreg: HOG + Logistic Regression (powerful approach)

    Args:
        model_name: str, one of the above models
        X_train: numpy array of shape (N, H, W) or (N, H, W, 3)
        y_train: numpy array of shape (N,)
        X_test: numpy array of shape (M, H, W) or (M, H, W, 3)
        y_test: numpy array of shape (M,)

    Returns:
        dict containing results
    """
    print("ℹ️  Traditional methods use CPU (scikit-learn is CPU-only)")
    results = {}

    # Step 1: Extract features
    print("\n=== Feature Extraction ===")
    feature_start = time.time()

    if model_name.startswith("hist_"):
        # Simple color histogram features
        print("Using Color Histogram features (simple baseline)...")
        X_train_features = extract_color_histogram(X_train, bins=32)
        X_test_features = extract_color_histogram(X_test, bins=32)
    elif model_name.startswith("hog_"):
        # HOG features (powerful approach)
        print("Using HOG features (powerful keypoint-based approach)...")
        X_train_features = extract_hog_features(X_train)
        X_test_features = extract_hog_features(X_test)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    feature_time = time.time() - feature_start
    print(f"Feature extraction completed in {feature_time:.2f}s")
    print(f"Feature shape: {X_train_features.shape}")

    # Step 2: Train classifier
    print("\n=== Training Classifier ===")
    train_start = time.time()

    if model_name.endswith("_svm"):
        print("Training SVM with RBF kernel...")
        print("(This may take a while, SVM doesn't provide progress updates)")
        classifier = SVC(kernel='rbf', C=10.0, gamma='scale', verbose=False)
        classifier.fit(X_train_features, y_train)
        print("✓ SVM training completed")
    elif model_name.endswith("_logreg"):
        print("Training Logistic Regression...")
        # Removed deprecated multi_class parameter (it defaults to 'auto' which uses multinomial for lbfgs)
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs', verbose=0)
        classifier.fit(X_train_features, y_train)
        print("✓ Logistic Regression training completed")
    else:
        raise ValueError(f"Unknown classifier in model: {model_name}")

    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f}s")

    # Step 3: Make predictions
    print("\n=== Evaluation ===")
    test_start = time.time()

    print("Making predictions on test set...")
    y_pred = classifier.predict(X_test_features)

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