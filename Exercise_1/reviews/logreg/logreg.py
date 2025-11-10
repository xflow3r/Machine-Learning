import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from Exercise_1.preprocess_datasets import load_amazon_review_dataset

# Load data (already numeric features)
x_train, x_test, y_train, y_test = load_amazon_review_dataset()

# Make y a 1-D vector
y_train = np.asarray(y_train).ravel()

# Scale (works for dense; if your matrices are sparse, use StandardScaler(with_mean=False))
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Training...")
# Logistic Regression with class balancing; tune C via CV
logreg = LogisticRegression(
    max_iter=1000,
    class_weight=None,
    solver="liblinear"  # use 'saga' if your data is sparse and large
)

param_grid = {"C": np.logspace(-4, 2, 7)}  # 0.01 ... 100
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    logreg,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=10,
    refit=True
)

grid.fit(x_train, y_train)
print(f"Best C: {grid.best_params_['C']:.3g}")
print(f"CV (f1_macro): {grid.best_score_:.4f}")

# Predict on test
y_pred = grid.predict(x_test)
print(y_pred[:10])

# Compare distributions between training labels and predicted ones
plt.figure(figsize=(8, 5))
plt.hist(y_train, bins=len(set(y_train)), alpha=0.6, label="Train (y_train)")
plt.hist(y_pred,  bins=len(set(y_pred)),  alpha=0.6, label="Predicted (y_pred)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Distribution Comparison: Training vs Predicted Classes (LogReg)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Save predictions with test IDs
test_ids = pd.read_csv('../../Datasets/amazon_review_test.csv')["ID"]
output = pd.DataFrame({"ID": test_ids, "Class": y_pred})
output.to_csv("./logreg_voting_result.csv", index=False)