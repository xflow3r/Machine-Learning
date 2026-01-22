import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy and F1-macro score

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary containing accuracy and f1_macro
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save confusion matrix

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {save_path}")