import time
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Your existing import ---
from Exercise_1.preprocess_datasets import load_amazon_review_dataset

RANDOM_STATE = 2742


def build_pipeline(n_neighbors=1, weights="uniform", p=2):
    """
    Standardized KNN in a single pipeline (prevents leakage in CV).
    """
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p))
    ])


def evaluate_holdout(X, y, n_neighbors=1, holdout_pct=0.2, weights="uniform", p=2):
    """
    Train/validate once with a stratified holdout split.
    Reports timing, accuracy, confusion matrix and a short report.
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

    t1 = time.time()
    y_pred_val = pipe.predict(X_val)
    infer_time = time.time() - t1

    val_acc = accuracy_score(y_val, y_pred_val)
    cm = confusion_matrix(y_val, y_pred_val)

    return {
        "method": f"Holdout ({int((1-holdout_pct)*100)}/{int(holdout_pct*100)})",
        "params": {"n_neighbors": n_neighbors, "weights": weights, "p": p},
        "model": pipe,
        "train_time": train_time,
        "infer_time": infer_time,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_val, y_pred_val, zero_division=0)
    }


def evaluate_cv(X, y, n_neighbors=1, n_folds=5, weights="uniform", p=2):
    """
    k-fold cross-validation over the whole training set with proper preprocessing in CV.
    """
    pipe = build_pipeline(n_neighbors=n_neighbors, weights=weights, p=p)

    t0 = time.time()
    cv = cross_validate(
        pipe, X, y,
        cv=n_folds,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
        n_jobs=-1
    )
    train_time = time.time() - t0

    # Fit final model on full train set for later test predictions
    pipe.fit(X, y)

    return {
        "method": f"{n_folds}-Fold CV",
        "params": {"n_neighbors": n_neighbors, "weights": weights, "p": p},
        "model": pipe,
        "train_time": train_time,
        "train_accuracy": float(np.mean(cv["train_accuracy"])),
        "val_accuracy": float(np.mean(cv["test_accuracy"])),
        "val_accuracy_std": float(np.std(cv["test_accuracy"])),
        "f1_macro": float(np.mean(cv["test_f1_macro"]))
    }


def pick_best(results):
    """
    Select by highest validation accuracy, breaking ties by std (lower is better) and training time (faster is better).
    """
    def key(r):
        return (r["val_accuracy"], -r.get("val_accuracy_std", 0.0), -r["train_time"])
    return max(results, key=key)


def main():
    # ---------------- Load data exactly as you had ----------------
    X_train, X_test, y_train, y_test = load_amazon_review_dataset()

    # ---------------- Experiments (holdout + CV over several ks) ----------------
    ks = [1, 3, 5, 7, 9]
    weightings = ["uniform", "distance"]  # optional: try both
    p_list = [2]  # Euclidean. Add 1 for Manhattan if you like.

    results = []

    print("=" * 100)
    print("Running HOLDOUT evaluations...")
    for k in ks:
        for w in weightings:
            for p in p_list:
                r = evaluate_holdout(X_train, y_train, n_neighbors=k, holdout_pct=0.2, weights=w, p=p)
                results.append(r)
                print(f"[Holdout] k={k}, weights={w}, p={p} | "
                      f"train_acc={r['train_accuracy']:.4f} val_acc={r['val_accuracy']:.4f} "
                      f"time: fit={r['train_time']:.3f}s infer={r['infer_time']:.3f}s")

    print("\nRunning CROSS-VALIDATION evaluations...")
    for k in ks:
        for w in weightings:
            for p in p_list:
                r = evaluate_cv(X_train, y_train, n_neighbors=k, n_folds=5, weights=w, p=p)
                results.append(r)
                print(f"[{r['method']}] k={k}, weights={w}, p={p} | "
                      f"train_acc={r['train_accuracy']:.4f} val_acc={r['val_accuracy']:.4f} "
                      f"(sd={r.get('val_accuracy_std', 0.0):.4f}) f1_macro={r.get('f1_macro', float('nan')):.4f} "
                      f"time: total={r['train_time']:.3f}s")

    # ---------------- Summarize results ----------------
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
            "Val Acc": f"{r['val_accuracy']:.4f}" + (f" (sd {r['val_accuracy_std']:.4f})" if "val_accuracy_std" in r else ""),
        })
    results_df = pd.DataFrame(rows).sort_values(by="Val Acc", ascending=False)
    print(results_df.to_string(index=False))

    # ---------------- Pick best and show extra diagnostics (holdout has CM/report) ----------------
    best = pick_best(results)
    print("\n" + "=" * 100)
    print(f"Best model: {best['method']} with params {best['params']}")
    print(f"Best validation accuracy: {best['val_accuracy']:.4f}")

    if "confusion_matrix" in best:
        print("\nConfusion Matrix (Best Holdout Model):")
        print(best["confusion_matrix"])
        print("\nPer-class report:")
        print(best["classification_report"])

    # ---------------- Fit best on full training set and predict test ----------------
    # If best came from holdout, refit identical spec on full training set to use all data.
    best_spec = best["params"]
    final_model = build_pipeline(**best_spec)
    t0 = time.time()
    final_model.fit(X_train, y_train)
    final_fit_time = time.time() - t0
    print(f"\nFinal training on full data: {final_fit_time:.3f}s")

    y_pred = final_model.predict(X_test)
    print("\nFirst 30 predictions:")
    print(y_pred[:30])

    # ---------------- Plot distribution comparison (train labels vs predictions) ----------------
    plt.figure(figsize=(8, 5))
    plt.hist(y_train, bins=len(set(y_train)), alpha=0.6, label="Train (y_train)")
    plt.hist(y_pred,  bins=len(set(y_pred)),  alpha=0.6, label="Predicted (y_pred)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution Comparison: Training vs Predicted Classes")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # ---------------- Save predictions as before ----------------
    test_ids = pd.read_csv('../../Datasets/amazon_review_test.csv')["ID"]
    output = pd.DataFrame({"ID": test_ids, "Class": y_pred})
    out_path = Path("./voting_knn_result.csv")
    output.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
