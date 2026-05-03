# feature_engineering.py - V1 Baseline Features
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib

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

    # ── 1. TF-IDF VECTORIZATION ───────────────────────────────────────────────

    vectorizer = TfidfVectorizer(
        max_features=10000,      # Keep top 10,000 most informative words
        ngram_range=(1, 2),      # Use single words AND two-word phrases (bigrams)
                                 # e.g., "verify account", "click here", "limited time"
        min_df=2,                # Ignore words that appear in fewer than 2 emails
        max_df=0.95,             # Ignore words that appear in more than 95% of emails
        sublinear_tf=True        # Apply log normalization to term frequency
    )

    X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
    print(f"TF-IDF matrix shape: {X_tfidf.shape}")
    # Shape will be: (num_emails, 10000)

    # ── 2. HANDCRAFTED PHISHING FEATURES ─────────────────────────────────────
    # These explicit features help the model catch patterns TF-IDF might miss

    SUSPICIOUS_WORDS = [
        'urgent', 'verify', 'login', 'click', 'account', 'suspended',
        'password', 'confirm', 'bank', 'credit', 'update', 'immediately',
        'expire', 'winner', 'congratulations', 'free', 'limited', 'offer',
        'security', 'alert', 'unusual', 'activity', 'action required'
    ]

    def extract_custom_features(text):
        """
        Extract phishing-specific signals from raw email text.
        Returns a feature vector (list of numbers).
        """
        features = {}

        # Feature 1: Count of suspicious words present
        text_lower = text.lower() if isinstance(text, str) else ""
        suspicious_count = sum(1 for word in SUSPICIOUS_WORDS if word in text_lower)
        features['suspicious_word_count'] = suspicious_count

        # Feature 2: Number of URLs
        url_count = len(re.findall(r'http\S+|www\S+|https\S+', text_lower))
        features['url_count'] = url_count

        # Feature 3: Number of email addresses mentioned
        email_count = len(re.findall(r'\S+@\S+', text_lower))
        features['email_address_count'] = email_count

        # Feature 4: Presence of urgency language (binary)
        urgency_words = ['urgent', 'immediately', 'expire', 'action required', 'limited time']
        features['has_urgency'] = int(any(w in text_lower for w in urgency_words))

        # Feature 5: Excessive capitalization (shouting = phishing signal)
        if isinstance(text, str) and len(text) > 0:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
        else:
            caps_ratio = 0
        features['caps_ratio'] = caps_ratio

        # Feature 6: Text length (phishing emails are often short and punchy)
        features['text_length'] = len(text_lower)

        # Feature 7: Exclamation mark count
        features['exclamation_count'] = text_lower.count('!')

        return list(features.values())

    # Apply to the ORIGINAL (uncleaned) text to preserve URLs and capitalization
    print("Extracting custom features...")
    custom_features = np.array(df['text'].apply(extract_custom_features).tolist())
    print(f"Custom features shape: {custom_features.shape}")

    # ── 3. COMBINE TF-IDF + CUSTOM FEATURES ──────────────────────────────────
    # Horizontally stack: [TF-IDF matrix | custom features]
    X_custom_sparse = sp.csr_matrix(custom_features)
    X_combined = sp.hstack([X_tfidf, X_custom_sparse])
    print(f"Combined feature matrix shape: {X_combined.shape}")

    y = df['label'].values

    # ── 4. SAVE ARTIFACTS ─────────────────────────────────────────────────────
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    sp.save_npz(DATA_DIR / "X_features.npz", X_combined)
    np.save(DATA_DIR / "y_labels.npy", y)
    print("Vectorizer and features saved.")


if __name__ == "__main__":
    main()
