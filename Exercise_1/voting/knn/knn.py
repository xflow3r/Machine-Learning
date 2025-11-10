import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer   # or: from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
import numpy as np

from Exercise_1.preprocess_datasets import load_voting_dataset

# Load data
x_train, x_test, y_train, y_test = load_voting_dataset()

# Build a pipeline that handles NaNs, scaling, then KNN
# If all features are numeric, median imputation is a solid default.
# Swap SimpleImputer(...) with KNNImputer(...) if you want a fancier imputation.
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),      # or KNNImputer(n_neighbors=5, weights="distance")
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=7)
)

# Fit + predict
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
print(y_pred[:30])

# Compare distributions between training labels and predicted ones
plt.figure(figsize=(8, 5))
plt.hist(y_train, bins=len(np.unique(y_train)), alpha=0.6, label="Train (y_train)")
plt.hist(y_pred,  bins=len(np.unique(y_pred)),  alpha=0.6, label="Predicted (y_pred)")
plt.xlabel("class")
plt.ylabel("Count")
plt.title("Distribution Comparison: Training vs Predicted Classes")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Create Kaggle submission with original test IDs and predicted class
test_ids = pd.read_csv('../../Datasets/voting_test.csv')["ID"]
output = pd.DataFrame({"ID": test_ids, "Class": y_pred})
output.to_csv("./voting_knn_result.csv", index=False)
