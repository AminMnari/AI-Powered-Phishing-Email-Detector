"""
Predictor v2 Improved: Uses refined features + v2 model for better precision on real-world emails.
This version includes legitimacy signals to counteract false positives.
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
THRESHOLD = 0.65


def _load_artifacts_if_needed():
    global vectorizer, model

    if vectorizer is not None and model is not None:
        return

    vec_path = MODELS_DIR / "vectorizer_v2.pkl"
    model_path = MODELS_DIR / "model_v2.pkl"

    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError("V2 artifacts not found. Run: python src/feature_engineering_v2.py then python src/model_training_v2.py")

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)


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


def extract_improved_features(text_raw, text_cleaned):
    text_raw_lower = text_raw.lower() if isinstance(text_raw, str) else ""
    text_clean_lower = text_cleaned.lower() if isinstance(text_cleaned, str) else ""

    PHISHING_KEYWORDS = [
        "verify account", "confirm identity", "validate credentials", "unusual activity", "suspicious activity",
        "account suspended", "account locked", "login attempt", "unauthorized access", "compromised", "click here",
        "click now", "act now", "limited time", "expire", "urgent action", "update payment", "billing problem",
        "confirm payment", "credit card", "bank account", "wire transfer", "winner", "congratulations",
        "claim reward", "free gift", "you won",
    ]

    LEGITIMATE_KEYWORDS = [
        "regards", "best regards", "respectfully", "please see attached", "agenda", "meeting notes",
        "team", "team member", "colleague", "department", "analyst", "manager", "engineer", "director",
    ]

    features = {}
    features["phishing_keyword_count"] = sum(1 for kw in PHISHING_KEYWORDS if kw in text_raw_lower)
    features["legitimate_keyword_count"] = sum(1 for kw in LEGITIMATE_KEYWORDS if kw in text_raw_lower)
    features["url_count"] = len(re.findall(r"http\S+|www\S+|https\S+", text_raw_lower))
    features["email_address_count"] = len(re.findall(r"\S+@\S+", text_raw_lower))
    features["caps_ratio"] = sum(1 for c in text_raw if c.isupper()) / len(text_raw) if isinstance(text_raw, str) and len(text_raw) > 0 else 0.0
    features["text_length"] = len(text_clean_lower)
    features["exclamation_count"] = text_raw_lower.count("!")
    features["has_generic_greeting"] = int(any(g in text_raw_lower for g in ["dear sir", "dear madam", "dear friend", "dear user"]))
    features["has_signature"] = int(any(sig in text_raw_lower for sig in ["regards", "sincerely", "best", "thanks", "thank you"]))
    return list(features.values())


def get_detected_phishing_patterns(text):
    text_lower = text.lower() if isinstance(text, str) else ""
    PHISHING_KEYWORDS = [
        "verify account", "confirm identity", "validate credentials", "unusual activity", "suspicious activity",
        "account suspended", "account locked", "login attempt", "unauthorized access", "compromised", "click here",
        "click now", "act now", "limited time", "expire", "urgent action",
    ]
    return [kw for kw in PHISHING_KEYWORDS if kw in text_lower]


def predict_email(raw_email_text, threshold=None):
    if threshold is None:
        threshold = THRESHOLD

    _load_artifacts_if_needed()
    cleaned = clean_email(raw_email_text)
    tfidf_features = vectorizer.transform([cleaned])

    custom = np.array([extract_improved_features(raw_email_text, cleaned)])
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
        "decision_threshold": threshold,
        "detected_phishing_patterns": get_detected_phishing_patterns(raw_email_text),
        "url_count": len(re.findall(r"http\S+|www\S+|https\S+", raw_email_text or "")),
    }
