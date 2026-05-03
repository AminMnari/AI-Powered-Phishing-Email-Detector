"""
Improved feature engineering with domain-specific phishing indicators and legitimacy signals.
Use instead of feature_engineering.py for better precision on real-world emails.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    df = pd.read_csv(DATA_DIR / "cleaned_emails.csv")

    required_cols = {"text", "cleaned_text", "label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"cleaned_emails.csv is missing required columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["cleaned_text", "text", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_tfidf = vectorizer.fit_transform(df["cleaned_text"])
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")

    PHISHING_KEYWORDS = [
        "verify account", "confirm identity", "validate credentials",
        "unusual activity", "suspicious activity", "account suspended",
        "account locked", "login attempt", "unauthorized access", "compromised",
        "click here", "click now", "act now", "limited time", "expire", "urgent action",
        "update payment", "billing problem", "confirm payment", "credit card",
        "bank account", "wire transfer", "winner", "congratulations", "claim reward",
        "free gift", "you won",
    ]

    LEGITIMATE_KEYWORDS = [
        "regards", "best regards", "respectfully", "please see attached",
        "agenda", "meeting notes", "team", "team member", "colleague",
        "department", "analyst", "manager", "engineer", "director", "hr",
    ]

    def extract_improved_features(text_raw, text_cleaned):
        text_raw_lower = text_raw.lower() if isinstance(text_raw, str) else ""
        text_clean_lower = text_cleaned.lower() if isinstance(text_cleaned, str) else ""
        features = {}

        features["phishing_keyword_count"] = sum(1 for keyword in PHISHING_KEYWORDS if keyword in text_raw_lower)
        features["legitimate_keyword_count"] = sum(1 for keyword in LEGITIMATE_KEYWORDS if keyword in text_raw_lower)
        features["url_count"] = len(re.findall(r"http\S+|www\S+|https\S+", text_raw_lower))
        features["email_address_count"] = len(re.findall(r"\S+@\S+", text_raw_lower))
        features["caps_ratio"] = sum(1 for c in text_raw if c.isupper()) / len(text_raw) if isinstance(text_raw, str) and len(text_raw) > 0 else 0.0
        features["text_length"] = len(text_clean_lower)
        features["exclamation_count"] = text_raw_lower.count("!")
        features["has_generic_greeting"] = int(any(g in text_raw_lower for g in ["dear sir", "dear madam", "dear friend", "dear user"]))
        features["has_signature"] = int(any(sig in text_raw_lower for sig in ["regards", "sincerely", "best", "thanks", "thank you"]))
        return list(features.values())

    print("Extracting improved custom features...")
    custom_features = np.array(
        [extract_improved_features(text_raw, text_clean) for text_raw, text_clean in zip(df["text"], df["cleaned_text"])]
    )
    print(f"Custom features shape: {custom_features.shape}")

    X_custom_sparse = sp.csr_matrix(custom_features)
    X_combined = sp.hstack([X_tfidf, X_custom_sparse])
    print(f"Combined feature matrix shape: {X_combined.shape}")

    y = df["label"].values

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, MODELS_DIR / "vectorizer_v2.pkl")
    sp.save_npz(DATA_DIR / "X_features_v2.npz", X_combined)
    np.save(DATA_DIR / "y_labels_v2.npy", y)

    print("\nImproved features saved:")
    print("  - vectorizer_v2.pkl")
    print("  - X_features_v2.npz")
    print("  - y_labels_v2.npy")


if __name__ == "__main__":
    main()
