"""
PERSON B: Implement deep learning methods here

Required function signature:
    run_deep(model_name, train_loader, test_loader, device='cuda') -> dict

Args:
    model_name: str, one of {"cnn_small", "cnn_medium"}
    train_loader: PyTorch DataLoader for training
    test_loader: PyTorch DataLoader for testing
    device: str, 'cuda' or 'cpu'

Returns:
    dict containing:
        - 'accuracy': float
        - 'f1_macro': float
        - 'train_time': float (seconds)
        - 'test_time': float (seconds)
        - 'y_true': numpy array (ground truth labels)
        - 'y_pred': numpy array (predicted labels)
        - 'model_name': str
"""

import torch
import torch.nn as nn


def run_deep(model_name, train_loader, test_loader, device='cuda'):
    """
    Run deep learning methods (CNNs)

    TODO for Person B:
    1. Implement CNN-small architecture
    2. Implement CNN-medium architecture
    3. Implement training loop with timing
    4. Implement evaluation with timing
    5. Return results in the required format
    """
    raise NotImplementedError("Person B needs to implement this function")