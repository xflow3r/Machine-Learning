import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from fashion_mnist_downloader import download_fashion_mnist


def load_fashion_mnist(normalize=True):
    """
    Load Fashion-MNIST and return both numpy arrays and PyTorch dataloaders

    Returns:
        tuple: (X_train, y_train, X_test, y_test, train_loader, test_loader, class_names)
    """
    train_images, train_labels, test_images, test_labels, class_names = download_fashion_mnist()

    # Convert to float and normalize if requested
    X_train = train_images.astype(np.float32)
    X_test = test_images.astype(np.float32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    y_train = train_labels
    y_test = test_labels

    return X_train, y_train, X_test, y_test, class_names


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """
    Create PyTorch dataloaders from numpy arrays
    """
    # Add channel dimension for CNNs
    X_train_torch = torch.FloatTensor(X_train).unsqueeze(1)
    X_test_torch = torch.FloatTensor(X_test).unsqueeze(1)
    y_train_torch = torch.LongTensor(y_train)
    y_test_torch = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader