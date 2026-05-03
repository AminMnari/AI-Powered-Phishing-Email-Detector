"""
V2 Model Training: Uses refined features + class weights to reduce false positives.
Run: python src/feature_engineering_v2.py, then python src/model_training_v2.py
"""
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False


def evaluate_model(model, x_test, y_test, model_name: str, results_dir: Path) -> float:
    y_pred = model.predict(x_test)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} Results (V2 Features)")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    if HAS_PLOTTING:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
        plt.title(f"Confusion Matrix - {model_name} (V2)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plot_name = f"confusion_matrix_{model_name.replace(' ', '_')}_v2.png"
        plt.savefig(results_dir / plot_name)
        plt.close()

    return accuracy_score(y_test, y_pred)


def main():
    # Get project root (parent of src directory)
    PROJECT_ROOT = Path(__file__).parent.parent
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"

    x_file = data_dir / "X_features_v2.npz"
    y_file = data_dir / "y_labels_v2.npy"

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError("Missing V2 features. Run feature_engineering_v2.py first.")

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    x = sp.load_npz(x_file)
    y = np.load(y_file)

    print("Training V2 Model with Improved Features")
    print("=" * 50)
    print(f"Features: {x.shape}, Labels: {y.shape}")
    print(f"Phishing: {int(y.sum())}, Legitimate: {int((y == 0).sum())}")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining samples: {x_train.shape[0]}")
    print(f"Testing samples:  {x_test.shape[0]}")

    nb_model = ComplementNB()
    nb_model.fit(x_train, y_train)
    nb_acc = evaluate_model(nb_model, x_test, y_test, "Complement Naive Bayes", results_dir)

    lr_model = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs", random_state=42, class_weight="balanced")
    lr_model.fit(x_train, y_train)
    lr_acc = evaluate_model(lr_model, x_test, y_test, "Logistic Regression (Balanced)", results_dir)

    print(f"\n{'=' * 50}")
    print(f"Naive Bayes Accuracy:            {nb_acc:.4f}")
    print(f"Logistic Regression Accuracy:    {lr_acc:.4f}")

    best_model = lr_model if lr_acc >= nb_acc else nb_model
    best_name = "Logistic Regression (V2)" if lr_acc >= nb_acc else "Naive Bayes (V2)"
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, models_dir / "model_v2.pkl")
    print("Model saved to models/model_v2.pkl")

    print("\n" + "=" * 50)
    print("Next: Use predictor_v2_improved.py or predictor_ensemble.py with model_v2.pkl")


if __name__ == "__main__":
    main()
