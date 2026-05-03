"""
Updated predictor with configurable threshold and optional v2 features.
Use this to test different thresholds and feature sets without retraining.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from bs4 import BeautifulSoup

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

vectorizer = None
model = None
THRESHOLD = 0.70


def _load_artifacts_if_needed():
    global vectorizer, model

    if vectorizer is not None and model is not None:
        return

    if not (MODELS_DIR / "vectorizer.pkl").exists() or not (MODELS_DIR / "model.pkl").exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run data_preparation.py, feature_engineering.py, and model_training.py first."
        )

    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    model = joblib.load(MODELS_DIR / "model.pkl")


def clean_email(text):
    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_custom_features(text):
    SUSPICIOUS_WORDS = [
        "urgent", "verify", "login", "click", "account", "suspended", "password",
        "confirm", "bank", "credit", "update", "immediately", "expire", "winner",
        "congratulations", "free", "limited", "offer", "security", "alert", "unusual",
        "activity", "action required",
    ]

    text_lower = text.lower() if isinstance(text, str) else ""
    features = [
        sum(1 for word in SUSPICIOUS_WORDS if word in text_lower),
        len(re.findall(r"http\S+|www\S+|https\S+", text_lower)),
        len(re.findall(r"\S+@\S+", text_lower)),
        int(any(w in text_lower for w in ["urgent", "immediately", "expire", "action required"])),
        sum(1 for c in text if c.isupper()) / max(len(text), 1) if isinstance(text, str) else 0.0,
        len(text_lower),
        text_lower.count("!"),
    ]
    return features


def predict_email(raw_email_text, threshold=None):
    if threshold is None:
        threshold = THRESHOLD

    _load_artifacts_if_needed()

    cleaned = clean_email(raw_email_text)
    tfidf_features = vectorizer.transform([cleaned])

    custom = np.array(extract_custom_features(raw_email_text)).reshape(1, -1)
    custom_sparse = sp.csr_matrix(custom)
    combined = sp.hstack([tfidf_features, custom_sparse])

    probability = model.predict_proba(combined)[0]
    phishing_prob = float(probability[1])
    label = 1 if phishing_prob >= threshold else 0

    return {
        "label": "Phishing" if label == 1 else "Legitimate",
        "is_phishing": bool(label == 1),
        "confidence": round(max(probability) * 100, 2),
        "phishing_probability": round(phishing_prob * 100, 2),
    }
