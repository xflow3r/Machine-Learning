import pandas as pd
import numpy as np

def preprocess_bikes():
    df = pd.read_csv('datasets/SeoulBikeData.csv', encoding='unicode_escape')
    
    df = df.drop('Date', axis=1)
    
    target = 'Rented Bike Count'
    y = df[target]
    X = df.drop(target, axis=1)
    
    categorical_cols = ['Seasons', 'Holiday', 'Functioning Day']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    df_processed = pd.concat([X, y], axis=1)
    
    df_processed.to_csv('datasets/preprocessed_bikes.csv', index=False)
    
    return df_processed


def preprocess_cars():
    
    df = pd.read_csv('datasets/used_car_price_dataset_extended.csv')
    
    target = 'price_usd'
    y = df[target]
    X = df.drop(target, axis=1)
    
    X['service_history'] = X['service_history'].fillna(X['service_history'].mode()[0])
    
    categorical_cols = ['fuel_type', 'brand', 'transmission', 'color', 'service_history', 'insurance_valid']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    df_processed = pd.concat([X, y], axis=1)
    
    df_processed.to_csv('datasets/preprocessed_cars.csv', index=False)
    
    return df_processed


def preprocess_houses():
    
    df = pd.read_csv('datasets/housing.csv')
    
    df = df.drop('Id', axis=1)
    
    high_missing_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    df = df.drop(high_missing_cols, axis=1)
    
    target = 'SalePrice'
    y = df[target]
    X = df.drop(target, axis=1)
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].mode()[0])
    
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    df_processed = pd.concat([X, y], axis=1)
    
    df_processed.to_csv('datasets/preprocessed_houses.csv', index=False)
    
    return df_processed


if __name__ == "__main__":

    try:
        preprocess_bikes()
    except Exception as e:
        print(f"Error: {e}\n")
    
    try:
        preprocess_cars()
    except Exception as e:
        print(f"Error: {e}\n")
    
    try:
        preprocess_houses()
    except Exception as e:
        print(f"Error: {e}\n")
    
