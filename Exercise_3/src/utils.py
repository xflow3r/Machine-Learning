import random
import numpy as np
import torch
import os
import csv


def set_seed(seed=42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def save_results_to_csv(results_dict, csv_path='results/tables/results.csv'):
    """
    Save results to CSV file

    Args:
        results_dict: Dictionary containing results
        csv_path: Path to CSV file
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(results_dict)

    print(f"Results saved to {csv_path}")