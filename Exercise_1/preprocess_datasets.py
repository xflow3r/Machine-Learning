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

    # Preprocess: replace categorical values with numeric, keeping NaN for missing values
    train_feature_cols = df_train.columns.difference(["ID", "class"])  # features only
    df_train[train_feature_cols] = (
        df_train[train_feature_cols]
        .replace({"y": 1, "n": 0, "unknown": np.nan})
        .astype(float)  # Standard float64 naturally supports NaN
    )

    test_feature_cols = df_test.columns.difference(["ID"])  # test set has no 'class'
    df_test[test_feature_cols] = (
        df_test[test_feature_cols]
        .replace({"y": 1, "n": 0, "unknown": np.nan})
        .astype(float)  # Standard float64 naturally supports NaN
    )

    # Target and features
    y_train = df_train["class"]
    y_test = None  # No labels in test set
    x_train = df_train.drop(columns=["class"])
    x_test = df_test

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
    """Load Road Safety (OpenML 42803). Return x_train, x_test, y_train, y_test (target: Age_Band_of_Driver).
    
    Preprocessing applied:
    - Drop high-cardinality and uninformative columns
    - Convert Sex_of_Driver to numeric
    - Extract date features from Date column
    - Convert Time to time_period (encoded numerically)
    - Keep NaN values for other missing data
    """
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

    # PREPROCESSING STEPS

    # 1. Pare down columns for simpler model
    # Drop high-cardinality identifier and location columns, as well as columns with excessive missing values
    cols_to_drop = [
        'Age_of_Driver', # would be perfect predictor of Age_Band_of_Driver
        'Accident_Index',  # Unique identifier (no predictive value)
        'LSOA_of_Accident_Location',  # 25k+ unique locations (too high cardinality)
        '2nd_Road_Class',  # 48% missing
        'Junction_Control',  # 48% missing
        'Propulsion_Code',  # 23% missing
        'Casualty_IMD_Decile',  # 19% missing
        # Location coordinates
        'Location_Easting_OSGR',
        'Location_Northing_OSGR',
        'Longitude',
        'Latitude',
        '2nd_Road_Number',  # High cardinality + 13% missing
        '1st_Road_Number',  # High cardinality + 12% missing
        'Local_Authority_(District)',  # High cardinality (356 unique)
        'Local_Authority_(Highway)',  # High cardinality (204 unique)
    ]
    
    # Only drop columns that exist
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # 2. Convert Sex_of_Driver from string to numeric
    if 'Sex_of_Driver' in df.columns:
        # Convert '1.0', '2.0', '3.0' strings to numeric
        df['Sex_of_Driver'] = pd.to_numeric(df['Sex_of_Driver'], errors='coerce')
    
    # 3. Extract date features from Date column
    if 'Date' in df.columns:
        # Convert to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        # Extract features
        df['Date_Month'] = df['Date'].dt.month
        df['Date_Day'] = df['Date'].dt.day
        df['Date_Year'] = df['Date'].dt.year
        # Drop original Date column
        df = df.drop(columns=['Date'])
    
    # 4. Convert Time to time_period (morning/afternoon/evening/night) and encode numerically
    if 'Time' in df.columns:
        def time_to_period(time_str):
            """Convert time string to period of day (0=night, 1=morning, 2=afternoon, 3=evening)"""
            if pd.isna(time_str):
                return np.nan
            try:
                # Parse time string (e.g., "12:30")
                hour = int(time_str.split(':')[0])
                if 6 <= hour < 12:
                    return 1  # Morning
                elif 12 <= hour < 18:
                    return 2  # Afternoon
                elif 18 <= hour < 22:
                    return 3  # Evening
                else:
                    return 0  # Night (22-6)
            except (ValueError, AttributeError):
                return np.nan
        
        df['Time_Period'] = df['Time'].apply(time_to_period)
        df = df.drop(columns=['Time'])
    
    # All other missing values kept as NaN
    
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
