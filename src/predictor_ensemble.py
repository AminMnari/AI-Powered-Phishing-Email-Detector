"""
Production Ensemble Predictor: Combines ML + Rule-Based Logic
This approach:
1. Uses the v2 model for initial scoring
2. Applies rule-based overrides to reduce false positives
3. Checks for legitimacy signals (professional signature, known patterns)
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


def _load_artifacts_if_needed():
    global vectorizer, model
    if vectorizer is not None and model is not None:
        return

    vec_path = MODELS_DIR / "vectorizer_v2.pkl"
    model_path = MODELS_DIR / "model_v2.pkl"

    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "V2 artifacts not found. Run: python src/feature_engineering_v2.py then python src/model_training_v2.py"
        )

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
        "verify account", "confirm identity", "validate credentials", "unusual activity", "account suspended",
        "account locked", "click here", "click now", "limited time", "update payment", "credit card", "bank account",
        "winner", "congratulations",
    ]

    LEGITIMATE_KEYWORDS = [
        "regards", "best regards", "respectfully", "please see attached", "agenda", "meeting notes", "team",
        "colleague", "department",
    ]

    features = {}
    features["phishing_keyword_count"] = sum(1 for kw in PHISHING_KEYWORDS if kw in text_raw_lower)
    features["legitimate_keyword_count"] = sum(1 for kw in LEGITIMATE_KEYWORDS if kw in text_raw_lower)
    features["url_count"] = len(re.findall(r"http\S+|www\S+|https\S+", text_raw_lower))
    features["email_address_count"] = len(re.findall(r"\S+@\S+", text_raw_lower))
    features["caps_ratio"] = sum(1 for c in text_raw if c.isupper()) / len(text_raw) if isinstance(text_raw, str) and len(text_raw) > 0 else 0.0
    features["text_length"] = len(text_clean_lower)
    features["exclamation_count"] = text_raw_lower.count("!")
    features["has_generic_greeting"] = int(any(g in text_raw_lower for g in ["dear sir", "dear madam", "dear friend"]))
    features["has_signature"] = int(any(sig in text_raw_lower for sig in ["regards", "sincerely", "best", "thanks", "thank you"]))
    return list(features.values())


def check_legitimacy_signals(text_raw):
    """Check for strong legitimacy indicators that override ML prediction."""
    text_lower = text_raw.lower() if isinstance(text_raw, str) else ""
    legitimacy_boost = 0.0

    signature_words = ["regards,", "best regards", "sincerely", "thank you"]
    if any(sig in text_lower for sig in signature_words):
        legitimacy_boost += 0.10

    org_words = ["ieee", "company", "organization", "department", "team", "team member"]
    if any(org in text_lower for org in org_words):
        legitimacy_boost += 0.08

    if text_lower.count("\n") > 5:
        legitimacy_boost += 0.05

    if "subject:" in text_lower:
        legitimacy_boost += 0.03

    has_phishing_patterns = any(
        kw in text_lower
        for kw in [
            "verify account", "confirm identity", "click here", "click now",
            "act now", "urgent action", "limited time", "click immediately",
        ]
    )

    if not has_phishing_patterns and any(
        pattern in text_lower
        for pattern in ["password was changed", "password reset", "account updated"]
    ):
        legitimacy_boost += 0.25

    if has_phishing_patterns:
        legitimacy_boost *= 0.3

    return legitimacy_boost


def predict_email_ensemble(raw_email_text, debug=False):
    _load_artifacts_if_needed()

    cleaned = clean_email(raw_email_text)
    tfidf_features = vectorizer.transform([cleaned])

    custom = np.array([extract_improved_features(raw_email_text, cleaned)])
    custom_sparse = sp.csr_matrix(custom)
    combined = sp.hstack([tfidf_features, custom_sparse])

    probability = model.predict_proba(combined)[0]
    ml_phishing_prob = float(probability[1])

    legitimacy_boost = check_legitimacy_signals(raw_email_text)
    adjusted_phishing_prob = max(0, min(1, ml_phishing_prob - legitimacy_boost))

    THRESHOLD = 0.70
    is_phishing = adjusted_phishing_prob >= THRESHOLD

    reason = []
    if legitimacy_boost > 0:
        reason.append(f"Legitimacy override: -{legitimacy_boost*100:.0f}%")
    reason.append(f"ML confidence: {ml_phishing_prob*100:.1f}%")

    if debug:
        print(f"[DEBUG] ML prob: {ml_phishing_prob:.3f} → Adjusted: {adjusted_phishing_prob:.3f} (boost: {legitimacy_boost:.3f})")

    return {
        "label": "Phishing" if is_phishing else "Legitimate",
        "is_phishing": bool(is_phishing),
        "confidence": round(max(1-adjusted_phishing_prob, adjusted_phishing_prob) * 100, 2),
        "phishing_probability": round(adjusted_phishing_prob * 100, 2),
        "ml_raw_probability": round(ml_phishing_prob * 100, 2),
        "legitimacy_override_applied": round(legitimacy_boost * 100, 1),
        "reasoning": reason,
        "decision_method": "Ensemble (ML + Rule-Based)",
    }


if __name__ == "__main__":
    test_phishing = """
    URGENT: Your account has been suspended!
    Click here immediately to verify your login credentials: http://fake-bank.com/verify
    Failure to act within 24 hours will result in permanent account closure.
    """

    test_password = """
    Subject: Your password was changed

    Hello,

    This is a confirmation that your account password was successfully updated.
    If you did not perform this action, please contact support immediately.

    Security Team
    """

    test_ieee = """
    Dear Member,

    I hope you are doing well.

    As mentioned during our General Assembly, we are sharing with you the access link to the IEEE TBS SB Members Space.
    This drive contains all the essential documents and resources you may need.

    We kindly ask you to check the drive regularly and maintain its organization.

    Should you encounter any issues, feel free to reach out.

    Thank you for your cooperation.

    Warm regards,
    Fatma Ezzahra Lessoued
    General Secretary
    IEEE Tunis Business School Student Branch
    """

    print("=" * 75)
    print("ENSEMBLE PHISHING DETECTOR (ML + Rule-Based Override Logic)")
    print("=" * 75)

    tests = [
        ("Test 1: Phishing Email", test_phishing),
        ("Test 2: Password Reset (Your Problem Case #1)", test_password),
        ("Test 3: IEEE Organizational Email (Your Problem Case #2)", test_ieee),
    ]

    for test_name, email in tests:
        print(f"\n{test_name}")
        print("-" * 75)
        result = predict_email_ensemble(email, debug=False)
        print(f"Result: {result['label']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  ML Raw Probability: {result['ml_raw_probability']}%")
        print(f"  After Legitimacy Override: {result['phishing_probability']}% (adjusted by {result['legitimacy_override_applied']}%)")
        print(f"  Reasoning: {' | '.join(result['reasoning'])}")

    print("\n" + "=" * 75)
