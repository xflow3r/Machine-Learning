import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from fashion_mnist_downloader import download_fashion_mnist
from torchvision import datasets, transforms
import os


def load_fashion_mnist(normalize=True):
    """
    Load Fashion-MNIST and return numpy arrays

    Returns:
        tuple: (X_train, y_train, X_test, y_test, class_names)
    """
    train_images, train_labels, test_images, test_labels, class_names = download_fashion_mnist()

    X_train = train_images.astype(np.float32)
    X_test = test_images.astype(np.float32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    y_train = train_labels
    y_test = test_labels

    return X_train, y_train, X_test, y_test, class_names


def load_cifar10(normalize=True, data_dir='./cifar10_data'):
    """
    Load CIFAR-10 and return numpy arrays

    Returns:
        tuple: (X_train, y_train, X_test, y_test, class_names)
    """
    print(f"Loading CIFAR-10 dataset from {data_dir}...")

    # Download and load CIFAR-10
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)

    # Convert to numpy arrays
    X_train = train_dataset.data.astype(np.float32)  # Shape: (50000, 32, 32, 3)
    y_train = np.array(train_dataset.targets)

    X_test = test_dataset.data.astype(np.float32)  # Shape: (10000, 32, 32, 3)
    y_test = np.array(test_dataset.targets)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"CIFAR-10 loaded: Train={X_train.shape}, Test={X_test.shape}")

    return X_train, y_train, X_test, y_test, class_names


def load_dataset(dataset_name='fashion_mnist', normalize=True):
    """
    Load specified dataset

    Args:
        dataset_name: 'fashion_mnist' or 'cifar10'
        normalize: Whether to normalize to [0, 1]

    Returns:
        tuple: (X_train, y_train, X_test, y_test, class_names)
    """
    if dataset_name == 'fashion_mnist':
        return load_fashion_mnist(normalize=normalize)
    elif dataset_name == 'cifar10':
        return load_cifar10(normalize=normalize)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'fashion_mnist' or 'cifar10'")


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64,
                       augment=False, dataset_name='fashion_mnist'):
    """
    Create PyTorch dataloaders from numpy arrays with optional data augmentation

    Args:
        X_train, y_train, X_test, y_test: Data arrays
        batch_size: Batch size
        augment: Whether to use data augmentation
        dataset_name: Name of dataset for proper handling
    """
    # Determine if grayscale (Fashion-MNIST) or RGB (CIFAR-10)
    is_grayscale = (len(X_train.shape) == 3)  # (N, H, W)

    if is_grayscale:
        # Fashion-MNIST: Add channel dimension
        X_train_torch = torch.FloatTensor(X_train.copy()).unsqueeze(1)  # (N, 1, 28, 28)
        X_test_torch = torch.FloatTensor(X_test.copy()).unsqueeze(1)
    else:
        # CIFAR-10: Convert from (N, H, W, C) to (N, C, H, W)
        X_train_torch = torch.FloatTensor(X_train.copy()).permute(0, 3, 1, 2)
        X_test_torch = torch.FloatTensor(X_test.copy()).permute(0, 3, 1, 2)

    y_train_torch = torch.LongTensor(y_train.copy())
    y_test_torch = torch.LongTensor(y_test.copy())

    if augment:
        print("Creating dataloaders WITH data augmentation...")
        # Create augmented training dataset
        train_dataset = AugmentedDataset(X_train_torch, y_train_torch,
                                         is_grayscale=is_grayscale)
        test_dataset = TensorDataset(X_test_torch, y_test_torch)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=2)
    else:
        print("Creating dataloaders WITHOUT data augmentation...")
        train_dataset = TensorDataset(X_train_torch, y_train_torch)
        test_dataset = TensorDataset(X_test_torch, y_test_torch)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies data augmentation on-the-fly
    """

    def __init__(self, X, y, is_grayscale=True):
        self.X = X
        self.y = y
        self.is_grayscale = is_grayscale

        if is_grayscale:
            # Augmentation for grayscale images (Fashion-MNIST)
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            # Augmentation for RGB images (CIFAR-10)
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.y[idx]

        # Apply augmentation
        img = self.transform(img)

        return img, label