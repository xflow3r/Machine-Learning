#!/usr/bin/env python3
import openml
import pandas as pd

def main() -> None:
    dataset: openml.datasets.OpenMLDataset = openml.datasets.get_dataset(42803)
    df: pd.DataFrame
    df, *_ = dataset.get_data() #type: ignore
    print("--- DataFrame Info ---")
    print(df.info())
    print("\n--- First 5 Rows ---")
    print(df.head())
    print("\n--- Summary Statistics ---")
    print(df.describe(include='all'))
    print("Unique Accident_Index values:", df['Accident_Index'].nunique())

    # Casualty_Severity
    if 'Casualty_Severity' in df.columns:
        print("\nUnique Casualty_Severity values:", df['Casualty_Severity'].nunique())
        print("Values:", df['Casualty_Severity'].unique())
    else:
        print("Casualty_Severity column not found.")

    # Casualty_Class
    if 'Casualty_Class' in df.columns:
        print("\nUnique Casualty_Class values:", df['Casualty_Class'].nunique())
        print("Values:", df['Casualty_Class'].unique())
    else:
        print("Casualty_Class column not found.")

if __name__ == "__main__":
    main()
