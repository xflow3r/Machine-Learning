import pandas as pd
from scipy.io import arff


# There is nothing to preprocess here
def load_amazon_review():
    df1 = pd.read_csv('../Datasets/amazon_review_learn.csv')
    df2 = pd.read_csv('../Datasets/amazon_review_test.csv')

    df1.dropna(how='any')

    return df1, df2


# You can specify how large your train/test should be.
# "train_percentage" says, how much data is going to be used for training
def load_phishing_dataset(train_percentage = 0.8):
    data, meta = arff.loadarff('Datasets/phishing_data.arff')
    df = pd.DataFrame(data)

    df = df.apply(pd.to_numeric)

    target_mapping = {-1: 'Legitimate', 0: 'Suspicious', 1: 'Phishing'}
    df['Result_Label'] = df['Result'].map(target_mapping)

    eighty_percent = train_percentage * df.shape[0]
    train = df.loc[:eighty_percent-1, :]
    test = df.loc[eighty_percent:, :]

    print('phishing train shape:', train.shape)
    print('phishing test  shape:', test.shape)

    return train, test

load_phishing_dataset()