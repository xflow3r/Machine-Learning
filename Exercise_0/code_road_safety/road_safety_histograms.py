#!/usr/bin/env python3
import os
import openml
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# road-safety
OPENML_DATASET_ID = 42803

INPUT_ATTRS = [
    '1st_Point_of_Impact',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Light_Conditions',
    'Vehicle_Type',
    'Casualty_Severity',
    'Day_of_Week',
    'Car_Passenger'
]
TARGET_ATTR = 'Age_Band_of_Driver'

PLOT_OUTPUT_DIR = os.path.join(os.getcwd(), '..', 'images_road_safety')
if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)

def plot_histograms(df: pd.DataFrame) -> None:
    # Input attributes histograms
    for attr in INPUT_ATTRS:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=attr, data=df, hue=attr, palette="Set2", legend=False)
        plt.title(f'Histogram of {attr}')
        plt.tight_layout()
        plt.savefig(f'{PLOT_OUTPUT_DIR}/{attr}_histogram.png')

    # Target attribute histogram
    plt.figure(figsize=(8, 5))
    sns.countplot(x=TARGET_ATTR, data=df, hue=TARGET_ATTR, palette="Set2", legend=False)
    plt.title(f'Histogram of {TARGET_ATTR}')
    plt.tight_layout()
    plt.savefig(f'{PLOT_OUTPUT_DIR}/{TARGET_ATTR}_histogram.png')

    print(f'Histograms saved in directory: {PLOT_OUTPUT_DIR}')


def main() -> None:
    print('Downloading Road Safety dataset from OpenML...')
    dataset: openml.datasets.OpenMLDataset = openml.datasets.get_dataset(OPENML_DATASET_ID)
    df: pd.DataFrame
    df, *_ = dataset.get_data() # type: ignore
    plot_histograms(df)


if __name__ == '__main__':
    main()
