"""
PERSON B: Deep learning methods implementation
Implements CNN-small and CNN-medium architectures
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm


class CNNSmall(nn.Module):
    """
    Small CNN architecture for Fashion-MNIST
    ~100K parameters
    """
    def __init__(self, num_classes=10):
        super(CNNSmall, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14x64

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 14x14

        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNNMedium(nn.Module):
    """
    Medium CNN architecture for Fashion-MNIST
    ~300K parameters
    """
    def __init__(self, num_classes=10):
        super(CNNMedium, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 14x14

        # Conv block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 7x7

        # Conv block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # 3x3

        # Flatten
        x = x.view(-1, 128 * 3 * 3)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train the model

    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs

    Returns:
        Training time in seconds
    """
    model.train()
    train_start = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    train_time = time.time() - train_start
    return train_time


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model

    Args:
        model: PyTorch model
        test_loader: DataLoader for testing
        device: Device to evaluate on

    Returns:
        tuple: (y_true, y_pred, test_time)
    """
    model.eval()
    y_true = []
    y_pred = []

    test_start = time.time()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_time = time.time() - test_start

    return np.array(y_true), np.array(y_pred), test_time


def run_deep(model_name, train_loader, test_loader, device='cuda', epochs=10, lr=0.001):
    """
    Run deep learning methods (CNNs)

    Args:
        model_name: str, one of {"cnn_small", "cnn_medium"}
        train_loader: PyTorch DataLoader for training
        test_loader: PyTorch DataLoader for testing
        device: str, 'cuda' or 'cpu'
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        dict containing results
    """
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Initialize model
    print("\n=== Model Initialization ===")
    if model_name == "cnn_small":
        model = CNNSmall(num_classes=10)
        print("Initialized CNN-Small")
    elif model_name == "cnn_medium":
        model = CNNMedium(num_classes=10)
        print("Initialized CNN-Medium")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train
    print("\n=== Training ===")
    train_time = train_model(model, train_loader, criterion, optimizer, device, epochs)
    print(f"Training completed in {train_time:.2f}s")

    # Evaluate
    print("\n=== Evaluation ===")
    y_true, y_pred, test_time = evaluate_model(model, test_loader, device)
    print(f"Evaluation completed in {test_time:.2f}s")

    # Compile results
    results = {
        'accuracy': None,  # Will be computed in main.py
        'f1_macro': None,  # Will be computed in main.py
        'train_time': train_time,
        'test_time': test_time,
        'y_true': y_true,
        'y_pred': y_pred,
        'model_name': model_name
    }

    return results