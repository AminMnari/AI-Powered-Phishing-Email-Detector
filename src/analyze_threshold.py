"""
Analyze model predictions on test set and find optimal threshold for precision/recall tradeoff.
Run after model_training.py to inspect decision boundaries.
"""
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import auc, f1_score, precision_recall_curve, precision_score, recall_score

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def main():
    model = joblib.load(MODELS_DIR / "model.pkl")
    x_combined = sp.load_npz(DATA_DIR / "X_features.npz")
    y = np.load(DATA_DIR / "y_labels.npy")

    n = len(y)
    test_idx = int(0.8 * n)
    x_test = x_combined[test_idx:]
    y_test = y[test_idx:]

    y_proba = model.predict_proba(x_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    print("Threshold Analysis for Phishing Detector")
    print("=" * 60)
    print(f"Test set size: {len(y_test)}")
    print(f"Phishing ratio: {y_test.sum() / len(y_test):.2%}")
    print(f"PR-AUC: {pr_auc:.4f}\n")

    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        y_pred = (y_proba >= threshold).astype(int)
        precision_at_t = precision_score(y_test, y_pred, zero_division=0)
        recall_at_t = recall_score(y_test, y_pred, zero_division=0)
        f1_at_t = f1_score(y_test, y_pred, zero_division=0)

        marker = " ← default" if threshold == 0.50 else ""
        marker = " ← RECOMMENDED (reduce false positives)" if threshold == 0.70 else marker

        print(f"{threshold:<12.2f} {precision_at_t:<12.4f} {recall_at_t:<12.4f} {f1_at_t:<12.4f} {marker}")

    print("\n" + "=" * 60)
    print("Recommendation: Use threshold=0.70 to reduce false positives")
    print("This ensures higher confidence before flagging as phishing.")


if __name__ == "__main__":
    main()
