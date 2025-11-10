import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

from Exercise_1.preprocess_datasets import load_road_safety_dataset

RANDOM_STATE = 2742


# ---------- Model builder ----------
def build_pipeline(n_neighbors=7, weights="uniform", p=2, imputer="median"):
    """
    SimpleImputer + StandardScaler + KNN (all-numeric features).
    """
    if imputer == "median":
        imp = SimpleImputer(strategy="median")
    else:
        # placeholder for KNNImputer, if you later want to switch:
        # from sklearn.impute import KNNImputer
        # imp = KNNImputer(n_neighbors=5, weights="distance")
        imp = SimpleImputer(strategy="median")

    return Pipeline([
        ("imputer", imp),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)),
    ])


# ---------- Evaluation helpers ----------
def evaluate_holdout(X, y, n_neighbors=7, weights="uniform", p=2, holdout_pct=0.2):
    """
    Single stratified holdout with timings and diagnostics.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=holdout_pct, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(n_neighbors=n_neighbors, weights=weights, p=p)

    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred_tr = pipe.predict(X_tr)
    train_acc = accuracy_score(y_tr, y_pred_tr)
    train_f1m = f1_score(y_tr, y_pred_tr, average="macro", zero_division=0)

    t1 = time.time()
    y_pred_val = pipe.predict(X_val)
    infer_time = time.time() - t1

    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1m = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
    cm = confusion_matrix(y_val, y_pred_val)
    report = classification_report(y_val, y_pred_val, zero_division=0)

    return {
        "method": f"Holdout ({int((1-holdout_pct)*100)}/{int(holdout_pct*100)})",
        "params": {"n_neighbors": n_neighbors, "weights": weights, "p": p},
        "model": pipe,
        "train_time": train_time,
        "infer_time": infer_time,
        "train_accuracy": train_acc,
        "train_f1_macro": train_f1m,
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1m,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def evaluate_cv(X, y, n_neighbors=7, weights="uniform", p=2, n_folds=5):
    """
    k-fold cross-validation with preprocessing inside folds.
    """
    pipe = build_pipeline(n_neighbors=n_neighbors, weights=weights, p=p)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    t0 = time.time()
    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring={"acc": "accuracy", "f1m": "f1_macro"},
        return_train_score=True,
        n_jobs=-1
    )
    total_time = time.time() - t0

    # Fit a final model on the full training set for later test predictions
    pipe.fit(X, y)

    return {
        "method": f"{n_folds}-Fold CV",
        "params": {"n_neighbors": n_neighbors, "weights": weights, "p": p},
        "model": pipe,
        "train_time": total_time,
        "train_accuracy": float(np.mean(scores["train_acc"])),
        "train_f1_macro": float(np.mean(scores["train_f1m"])),
        "val_accuracy": float(np.mean(scores["test_acc"])),
        "val_accuracy_std": float(np.std(scores["test_acc"])),
        "val_f1_macro": float(np.mean(scores["test_f1m"])),
    }


def pick_best(results):
    """
    Choose by highest validation F1-macro, then accuracy, then lower variance, then faster time.
    """
    def key(r):
        return (
            r.get("val_f1_macro", -1.0),
            r.get("val_accuracy", -1.0),
            -r.get("val_accuracy_std", 0.0),
            -r["train_time"],
        )
    return max(results, key=key)


# ---------- Main experiment ----------
def main():
    # Load data
    x_train, x_test, y_train, y_test = load_road_safety_dataset()
    x_train = x_train[:100000]
    y_train = y_train[:100000]

    # Hyperparameter grid
    ks = [3, 5, 7, 9, 11]
    weightings = ["uniform", "distance"]
    p_list = [1, 2]  # Manhattan & Euclidean

    results = []

    print("=" * 100)
    print("Running HOLDOUT evaluations...")
    for k in ks:
        for w in weightings:
            for p in p_list:
                r = evaluate_holdout(x_train, y_train, n_neighbors=k, weights=w, p=p, holdout_pct=0.2)
                results.append(r)
                print(f"[Holdout] k={k:>2}, weights={w:<8}, p={p} | "
                      f"train_acc={r['train_accuracy']:.4f} train_f1m={r['train_f1_macro']:.4f} | "
                      f"val_acc={r['val_accuracy']:.4f} val_f1m={r['val_f1_macro']:.4f} | "
                      f"time: fit={r['train_time']:.3f}s infer={r['infer_time']:.3f}s")

    print("\nRunning CROSS-VALIDATION evaluations...")
    for k in ks:
        for w in weightings:
            for p in p_list:
                r = evaluate_cv(x_train, y_train, n_neighbors=k, weights=w, p=p, n_folds=5)
                results.append(r)
                print(f"[{r['method']}] k={k:>2}, weights={w:<8}, p={p} | "
                      f"train_acc={r['train_accuracy']:.4f} train_f1m={r['train_f1_macro']:.4f} | "
                      f"val_acc={r['val_accuracy']:.4f} (sd={r['val_accuracy_std']:.4f}) "
                      f"val_f1m={r['val_f1_macro']:.4f} | time={r['train_time']:.3f}s")

    # Summarize results
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY TABLE")
    rows = []
    for r in results:
        rows.append({
            "Method": r["method"],
            "k": r["params"]["n_neighbors"],
            "Weights": r["params"]["weights"],
            "p": r["params"]["p"],
            "Train Time (s)": f"{r['train_time']:.3f}",
            "Train Acc": f"{r['train_accuracy']:.4f}",
            "Train F1m": f"{r['train_f1_macro']:.4f}",
            "Val Acc": f"{r['val_accuracy']:.4f}" + (f" (sd {r['val_accuracy_std']:.4f})" if "val_accuracy_std" in r else ""),
            "Val F1m": f"{r.get('val_f1_macro', float('nan')):.4f}",
        })
    results_df = pd.DataFrame(rows).sort_values(by=["Val F1m", "Val Acc"], ascending=False)
    print(results_df.to_string(index=False))

    # Pick best spec and (if holdout) print diagnostics
    best = pick_best(results)
    print("\n" + "=" * 100)
    print(f"Best model: {best['method']} with params {best['params']}")
    print(f"Best validation: F1m={best.get('val_f1_macro', float('nan')):.4f} | Acc={best['val_accuracy']:.4f}")

    if "confusion_matrix" in best:
        print("\nConfusion Matrix (Best Holdout Model):")
        print(best["confusion_matrix"])
        print("\nPer-class report:")
        print(best["classification_report"])

    # Refit best spec on full TRAIN and predict TEST
    spec = best["params"]
    final_model = build_pipeline(**spec)
    t0 = time.time()
    final_model.fit(x_train, y_train)
    final_fit_time = time.time() - t0
    print(f"\nFinal training on full data: {final_fit_time:.3f}s")

    y_pred = final_model.predict(x_test)

    print("\nFirst 30 predictions vs truths:")
    print(y_pred[:30])
    print(np.array(y_test)[:30])

    # =========================
    # Plot: your original density heatmap (works when labels are numeric/ordinal)
    # =========================
    y_true = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    plt.figure(figsize=(7, 6))
    plt.hist2d(y_true, y_pred_arr, bins=30, cmap="YlOrBr", cmin=1)
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

    # Optional: confusion matrix heatmap when class count is modest (<= 20)
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    if unique_classes.size <= 20:
        # If labels aren’t integers/strings consistently, encode to align axes
        if not np.issubdtype(unique_classes.dtype, np.number):
            le = LabelEncoder().fit(unique_classes)
            y_true_cm = le.transform(y_test)
            y_pred_cm = le.transform(y_pred)
            class_names = le.classes_
        else:
            # sort class labels for stable axes
            class_names = np.sort(unique_classes)
            # map to index positions
            mapper = {c: i for i, c in enumerate(class_names)}
            y_true_cm = np.vectorize(mapper.get)(y_test)
            y_pred_cm = np.vectorize(mapper.get)(y_pred)

        cm = confusion_matrix(y_true_cm, y_pred_cm, labels=np.arange(len(class_names)))
        ticks = np.arange(len(class_names))

        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest", aspect="auto")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
        plt.colorbar(label="Count")
        plt.tight_layout()
        plt.show()

    # Optional: save predictions (no explicit ID column available)
    out = pd.DataFrame({"Predicted": y_pred})
    out_path = Path("./road_safety_knn_predictions.csv")
    out.to_csv(out_path, index_label="Row")
    print(f"\nPredictions saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
