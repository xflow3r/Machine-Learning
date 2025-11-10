import time
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from Exercise_1.preprocess_datasets import load_amazon_review_dataset

RANDOM_STATE = 2742


def build_pipeline(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000):
    """
    StandardScaler + LogisticRegression in one pipeline (prevents leakage in CV).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=C,
            solver=solver,            # lbfgs supports l2; good default for dense data
            penalty="l2",
            class_weight=class_weight,
            max_iter=max_iter,
            n_jobs=None
        ))
    ])


def evaluate_holdout(X, y, C=1.0):
    """
    Single stratified holdout. Reports timing, accuracy, macro-F1, CM, and class report.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(C=C)

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
        "method": "Holdout (80/20)",
        "params": {"C": C},
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


def evaluate_cv(X, y, C=1.0, n_folds=5):
    """
    k-fold CV for a fixed C with preprocessing inside the CV folds.
    """
    pipe = build_pipeline(C=C)

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

    # Fit a final model on the whole training set for later test predictions
    pipe.fit(X, y)

    return {
        "method": f"{n_folds}-Fold CV",
        "params": {"C": C},
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
    Select by highest validation F1-macro, then accuracy, then lower variance, then faster time.
    """
    def key(r):
        return (
            r.get("val_f1_macro", -1.0),
            r.get("val_accuracy", -1.0),
            -r.get("val_accuracy_std", 0.0),
            -r["train_time"],
        )
    return max(results, key=key)


def main():
    # ----- Load data -----
    X_train, X_test, y_train, y_test = load_amazon_review_dataset()
    y_train = np.asarray(y_train).ravel()

    # ----- Hyperparameter sweep over C (log grid) -----
    C_grid = np.logspace(-2, 2, 7)  # 0.01 ... 100

    results = []

    print("=" * 100)
    print("Running HOLDOUT evaluations...")
    for C in C_grid:
        r = evaluate_holdout(X_train, y_train, C=C)
        results.append(r)
        print(f"[Holdout] C={C:>7.3g} | "
              f"train_acc={r['train_accuracy']:.4f} train_f1m={r['train_f1_macro']:.4f} | "
              f"val_acc={r['val_accuracy']:.4f} val_f1m={r['val_f1_macro']:.4f} | "
              f"time: fit={r['train_time']:.3f}s infer={r['infer_time']:.3f}s")

    print("\nRunning CROSS-VALIDATION evaluations...")
    for C in C_grid:
        r = evaluate_cv(X_train, y_train, C=C, n_folds=5)
        results.append(r)
        print(f"[{r['method']}] C={C:>7.3g} | "
              f"train_acc={r['train_accuracy']:.4f} train_f1m={r['train_f1_macro']:.4f} | "
              f"val_acc={r['val_accuracy']:.4f} (sd={r['val_accuracy_std']:.4f}) "
              f"val_f1m={r['val_f1_macro']:.4f} | time={r['train_time']:.3f}s")

    # ----- Summarize results -----
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY TABLE")
    rows = []
    for r in results:
        rows.append({
            "Method": r["method"],
            "C": r["params"]["C"],
            "Train Time (s)": f"{r['train_time']:.3f}",
            "Train Acc": f"{r['train_accuracy']:.4f}",
            "Train F1m": f"{r['train_f1_macro']:.4f}",
            "Val Acc": f"{r['val_accuracy']:.4f}" + (f" (sd {r['val_accuracy_std']:.4f})" if "val_accuracy_std" in r else ""),
            "Val F1m": f"{r.get('val_f1_macro', float('nan')):.4f}",
        })
    results_df = pd.DataFrame(rows).sort_values(by=["Val F1m", "Val Acc"], ascending=False)
    print(results_df.to_string(index=False))

    # ----- Pick best spec and diagnose (holdout includes CM/report) -----
    best = pick_best(results)
    print("\n" + "=" * 100)
    print(f"Best model: {best['method']} with params {best['params']}")
    print(f"Best validation: F1m={best.get('val_f1_macro', float('nan')):.4f} | Acc={best['val_accuracy']:.4f}")

    if "confusion_matrix" in best:
        print("\nConfusion Matrix (Best Holdout Model):")
        print(best["confusion_matrix"])
        print("\nPer-class report:")
        print(best["classification_report"])

    # ----- Refit best spec on full training and predict test -----
    best_spec = best["params"]
    final_model = build_pipeline(**best_spec)
    t0 = time.time()
    final_model.fit(X_train, y_train)
    final_fit_time = time.time() - t0
    print(f"\nFinal training on full data: {final_fit_time:.3f}s")

    y_pred = final_model.predict(X_test)
    print("\nFirst 30 predictions:")
    print(y_pred[:30])

    # ----- Plot distribution comparison -----
    plt.figure(figsize=(8, 5))
    plt.hist(y_train, bins=len(set(y_train)), alpha=0.6, label="Train (y_train)")
    plt.hist(y_pred,  bins=len(set(y_pred)),  alpha=0.6, label="Predicted (y_pred)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution Comparison: Training vs Predicted Classes (LogReg)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # ----- Save predictions -----
    test_ids = pd.read_csv('../../Datasets/amazon_review_test.csv')["ID"]
    output = pd.DataFrame({"ID": test_ids, "Class": y_pred})
    out_path = Path("./logreg_voting_result.csv")
    output.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
