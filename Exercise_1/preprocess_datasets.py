import pandas as pd
from scipy.io import arff as arff
import numpy as np


# There is nothing to preprocess here
def load_amazon_review():
    df_train = pd.read_csv('Datasets/amazon_review_learn.csv')
    df_test = pd.read_csv('Datasets/amazon_review_test.csv')

    df_train.dropna(how="any", inplace=True)
    df_test.dropna(how="any", inplace=True)

    y = df_train["Class"]
    x_train = df_train.drop(columns=["Class"])

    print("amazon train shape:", x_train.shape)
    print("amazon test  shape:", df_test.shape)

    return x_train, df_test, y


def load_voting_dataset():
    df_train = pd.read_csv('Datasets/voting_learn.csv')
    df_test = pd.read_csv('Datasets/voting_test.csv')

    # Target and features
    y = df_train["class"]
    x_train = df_train.drop(columns=["ID", "class"])
    x_test = df_test.drop(columns=["ID"])

    # Convert to numeric
    for df in [x_train, x_test]:
        df.replace({"y": 1, "n": 0, "unknown": np.nan}, inplace=True)
        df.dropna(how="any", inplace=True)

    print("voting train shape:", x_train.shape)
    print("voting test  shape:", x_test.shape)

    return x_train, x_test, y


# You can specify how large your train/test should be.
# "train_percentage" says, how much data is going to be used for training
def load_phishing_dataset(train_percentage=0.8):
    data, meta = arff.loadarff('Datasets/phishing_data.arff')
    df = pd.DataFrame(data)

    df = df.apply(pd.to_numeric)

    target_mapping = {-1: 'Legitimate', 0: 'Suspicious', 1: 'Phishing'}
    df['Result_Label'] = df['Result'].map(target_mapping)

    eighty_percent = train_percentage * df.shape[0]
    train = df.loc[:eighty_percent - 1, :]
    test = df.loc[eighty_percent:, :]

    print('phishing train shape:', train.shape)
    print('phishing test  shape:', test.shape)

    return train, test
