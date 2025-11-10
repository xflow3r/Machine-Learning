import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

from Exercise_1.preprocess_datasets import load_phishing_dataset

# Load data
x_train, x_test, y_train, y_test = load_phishing_dataset()

# --- Preprocess: numeric vs categorical ---
numeric_sel = selector(dtype_include=np.number)
categorical_sel = selector(dtype_exclude=np.number)

numeric_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

categorical_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_sel),
        ("cat", categorical_pipe, categorical_sel),
    ]
)

# --- Model ---
clf = make_pipeline(
    preprocess,
    KNeighborsClassifier(n_neighbors=7)
)

# Fit + predict
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("y_pred[:10]:", y_pred[:10])
print("y_test[:10]:", y_test[:10])

# =========================
# Plot: predicted vs measured heatmap
# =========================
# Encode labels to integers for plotting
le = LabelEncoder()
le.fit(pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0))

y_true_enc = le.transform(y_test)
y_pred_enc = le.transform(y_pred)
class_names = le.classes_

plt.figure(figsize=(7, 6))
plt.hist2d(y_true_enc, y_pred_enc, bins=len(class_names), cmin=1)
# Perfect prediction diagonal
plt.plot([0, len(class_names)-1], [0, len(class_names)-1], "k-", lw=2)

plt.xlabel("Measured (y_test)")
plt.ylabel("Predicted (y_pred)")
plt.title("Predicted vs Measured (binned density)")

# Put class names on ticks
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)

cb = plt.colorbar()
cb.set_label("Number of samples")

plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# (Optional) Also show a standard confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred, labels=class_names)
plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest", aspect="auto")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks, class_names, rotation=45, ha="right")
plt.yticks(ticks, class_names)

# annotate counts
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

plt.colorbar(label="Count")
plt.tight_layout()
plt.show()
