import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from Exercise_1.preprocess_datasets import load_amazon_review_dataset

x_train, x_test, y_train, y_test = load_amazon_review_dataset()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print(y_pred[:30])

# Compare distributions between training labels and predicted ones
plt.figure(figsize=(8, 5))
plt.hist(y_train, bins=len(set(y_train)), alpha=0.6, label="Train (y_train)")
plt.hist(y_pred, bins=len(set(y_pred)), alpha=0.6, label="Predicted (y_pred)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Distribution Comparison: Training vs Predicted Classes")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Create a DataFrame with ID and predicted class
output = pd.DataFrame({
    "ID": pd.read_csv('../../Datasets/amazon_review_test.csv')["ID"],
    "Class": y_pred})

# Save to CSV (no index column)
output.to_csv("./voting_knn_result.csv", index=False)
