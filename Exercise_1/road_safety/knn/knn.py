import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Exercise_1.preprocess_datasets import load_road_safety_dataset

x_train, x_test, y_train, y_test = load_road_safety_dataset()

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
print(y_test[:30])


y_true = np.array(y_test)
y_pred = np.array(y_pred)

plt.figure(figsize=(7, 6))

# Draw a 2D histogram as background density
plt.hist2d(y_true, y_pred, bins=30, cmap="YlOrBr", cmin=1)

# Diagonal (perfect prediction line)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         'k-', lw=2)

plt.xlabel("Measured (y_test)")
plt.ylabel("Predicted (y_pred)")
plt.title("Predicted vs Measured with Density Heatmap")
plt.colorbar(label="Number of Samples")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
