#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.io import arff
import openml
from sklearn.model_selection import train_test_split
from pathlib import Path

# Opt in to pandas future behavior (no silent downcasting) to avoid FutureWarning on replace()
pd.set_option('future.no_silent_downcasting', True)

# Directory containing preprocess_datasets.py
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets"

amazon_train = DATASETS_DIR / "amazon_review_learn.csv"
amazon_test  = DATASETS_DIR / "amazon_review_test.csv"
voting_train = DATASETS_DIR / "voting_learn.csv"
voting_test  = DATASETS_DIR / "voting_test.csv"
phishing_data = DATASETS_DIR / "phishing_data.arff"
road_safety_id = 42803  # OpenML dataset ID for Road Safety

def load_amazon_review_dataset(debug=False):
    df_train = pd.read_csv(amazon_train)
    df_test = pd.read_csv(amazon_test)

    # Preprocess: drop rows with missing values
    for df in [df_train, df_test]:
        df.dropna(how="any", inplace=True)

    y_train = df_train["Class"]
    x_train = df_train.drop(columns=["Class"])
    y_test = None # No labels in test set
    x_test = df_test # no "Class" column in test set

    if debug:
        print("amazon x_train shape:", x_train.shape)
        print("amazon x_test  shape:", x_test.shape)
        print("amazon y_train shape:", y_train.shape)
        print("amazon y_test  :", y_test)

    return x_train, x_test, y_train, y_test


def load_voting_dataset(debug=False):
    df_train = pd.read_csv(voting_train)
    df_test = pd.read_csv(voting_test)

    # Preprocess: explicit replace and dtype cast to avoid future downcasting warnings
    train_feature_cols = df_train.columns.difference(["ID", "class"])  # features only
    df_train[train_feature_cols] = (
        df_train[train_feature_cols]
        .replace({"y": 1, "n": 0, "unknown": np.nan})
        .astype("Int64")
    )
    df_train.dropna(how="any", inplace=True)

    test_feature_cols = df_test.columns.difference(["ID"])  # test set has no 'class'
    df_test[test_feature_cols] = (
        df_test[test_feature_cols]
        .replace({"y": 1, "n": 0, "unknown": np.nan})
        .astype("Int64")
    )
    df_test.dropna(how="any", inplace=True)

    # Target and features
    y_train = df_train["class"]
    y_test = None  # No labels in test set
    x_train = df_train.drop(columns=["ID", "class"])
    x_test = df_test.drop(columns=["ID"])

    if debug:
        print("voting x_train shape:", x_train.shape)
        print("voting x_test  shape:", x_test.shape)
        print("voting y_train shape:", y_train.shape)
        print("voting y_test  :", y_test)

    return x_train, x_test, y_train, y_test


# You can specify how large your train/test should be.
# "train_percentage" says, how much data is going to be used for training
def load_phishing_dataset(train_percentage: float = 0.8, seed: int = 42, debug=False):
    data, meta = arff.loadarff(phishing_data)
    df = pd.DataFrame(data)

    df = df.apply(pd.to_numeric)

    target_mapping = {-1: 'Legitimate', 0: 'Suspicious', 1: 'Phishing'}
    df['Result_Label'] = df['Result'].map(target_mapping)

    # Drop missing rows to keep X/y aligned
    df.dropna(how='any', inplace=True)

    X = df.drop(columns=['Result'])
    y = df['Result']

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_percentage,
        random_state=seed,
        shuffle=True,
    )

    if debug:
        print('phishing X_train shape:', x_train.shape)
        print('phishing X_test  shape:', x_test.shape)
        print('phishing y_train size :', y_train.shape)
        print('phishing y_test  size :', y_test.shape)

    return x_train, x_test, y_train, y_test


def load_road_safety_dataset(train_percentage: float = 0.8, seed: int = 42, debug=False):
    """Load Road Safety (OpenML 42803). Return x_train, x_test, y_train, y_test (target: Age_Band_of_Driver)."""
    if not (0 < train_percentage <= 1):
        raise ValueError("train_percentage must be in (0,1].")

    # Fetch dataset (uses OpenML cache after first download)
    dataset = openml.datasets.get_dataset(road_safety_id)
    df, *_ = dataset.get_data()

    # Ensure DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)  # type: ignore

    target_col = 'Age_Band_of_Driver'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in Road Safety dataset.")

    # Drop rows missing the target
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_percentage,
        random_state=seed,
        shuffle=True,
    )

    if debug:
        print('road_safety X_train shape:', x_train.shape)
        print('road_safety X_test  shape:', x_test.shape)
        print('road_safety y_train size :', y_train.shape)
        print('road_safety y_test  size :', y_test.shape)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    try:
        load_amazon_review_dataset(debug=True)
        load_voting_dataset(debug=True)
        load_phishing_dataset(debug=True)
        load_road_safety_dataset(debug=True)
    except Exception as e:
        print(f"An error occurred during dataset loading: {e}")
    else:
        print("All datasets loaded successfully.")
