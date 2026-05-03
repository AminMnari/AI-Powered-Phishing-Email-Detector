import re
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from bs4 import BeautifulSoup

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"

vectorizer = None
model = None

SUSPICIOUS_WORDS = [
    "urgent",
    "verify",
    "login",
    "click",
    "account",
    "suspended",
    "password",
    "confirm",
    "bank",
    "credit",
    "update",
    "immediately",
    "expire",
    "winner",
    "congratulations",
    "free",
    "limited",
    "offer",
    "security",
    "alert",
    "unusual",
    "activity",
    "action required",
]


def _load_artifacts_if_needed() -> None:
    global vectorizer, model

    if vectorizer is not None and model is not None:
        return

    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run data_preparation.py, feature_engineering.py, and model_training.py first."
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)


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


def get_suspicious_words_found(text):
    text_lower = text.lower() if isinstance(text, str) else ""
    return [word for word in SUSPICIOUS_WORDS if word in text_lower]


def predict_email(raw_email_text):
    _load_artifacts_if_needed()

    cleaned = clean_email(raw_email_text)
    tfidf_features = vectorizer.transform([cleaned])

    custom = np.array(extract_custom_features(raw_email_text)).reshape(1, -1)
    custom_sparse = sp.csr_matrix(custom)
    combined = sp.hstack([tfidf_features, custom_sparse])

    label = model.predict(combined)[0]
    probability = model.predict_proba(combined)[0]

    return {
        "label": "Phishing" if label == 1 else "Legitimate",
        "is_phishing": bool(label == 1),
        "confidence": round(float(max(probability)) * 100, 2),
        "phishing_probability": round(float(probability[1]) * 100, 2),
        "suspicious_words_found": get_suspicious_words_found(raw_email_text),
        "url_count": len(re.findall(r"http\S+|www\S+|https\S+", raw_email_text or "")),
    }


if __name__ == "__main__":
    test_phishing = """
    URGENT: Your account has been suspended!
    Click here immediately to verify your login credentials: http://fake-bank.com/verify
    Failure to act within 24 hours will result in permanent account closure.
    """

    test_legit = """
    Hi Sarah, just wanted to follow up on the meeting notes from Tuesday.
    I've attached the slides for your review. Let me know if you have any questions.
    Best, John
    """

    print("=== Phishing Email Test ===")
    result = predict_email(test_phishing)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== Legitimate Email Test ===")
    result = predict_email(test_legit)
    for k, v in result.items():
        print(f"  {k}: {v}")
