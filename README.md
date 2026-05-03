# AI-Powered Phishing Email Detector

This project builds a binary email classifier:

- `1` = phishing/spam (malicious)
- `0` = legitimate/ham (safe)

It combines two datasets:

- Phishing and Legitimate Emails Dataset (synthetic)
- SpamAssassin Email Dataset (real-world)

## Project Structure

```
├── src/                               # All Python source modules
│   ├── data_preparation.py            # Load, normalize, and merge datasets
│   ├── feature_engineering.py         # V1: TF-IDF + 7 handcrafted features
│   ├── feature_engineering_v2.py      # V2: TF-IDF + 9 refined features + legitimacy signals
│   ├── model_training.py              # V1: Train NB + LR on v1 features
│   ├── model_training_v2.py           # V2: Train NB + LR on v2 features (class_weight='balanced')
│   ├── predictor.py                   # V1 Inference: uses v1 features + v1 model
│   ├── predictor_v2.py                # V2 Inference: uses v1 features + v2 model (threshold tuning)
│   ├── predictor_v2_improved.py       # V2 Improved: uses v2 features + v2 model
│   ├── predictor_ensemble.py          # Production: v2 model + rule-based legitimacy override
│   ├── app.py                         # Flask REST API + web server
│   ├── analyze_threshold.py           # Find optimal decision threshold
│   └── generate_project_report_docx.py # Generate academic DOCX report
│
├── tests/                             # Test suite
│   ├── test_api.py                    # Flask endpoint validation (password reset, phishing, org email)
│   └── test_v2_threshold.py           # Threshold tuning demonstration on v2 model
│
├── data/                              # Datasets and generated features
│   ├── phishing_legitimate.csv        # Input: 10,000 synthetic emails
│   ├── spam_assassin.csv              # Input: 5,796 real-world emails
│   ├── cleaned_emails.csv             # Output: merged + normalized (15,796 emails)
│   ├── X_features.npz                 # V1 features (sparse TF-IDF + 7 custom)
│   ├── y_labels.npy                   # V1 labels
│   ├── X_features_v2.npz              # V2 features (sparse TF-IDF + 9 custom)
│   └── y_labels_v2.npy                # V2 labels
│
├── models/                            # Trained model artifacts
│   ├── vectorizer.pkl                 # V1 TF-IDF vectorizer
│   ├── model.pkl                      # V1 trained Logistic Regression
│   ├── vectorizer_v2.pkl              # V2 TF-IDF vectorizer
│   └── model_v2.pkl                   # V2 trained Logistic Regression (balanced)
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis (data distribution, feature analysis)
│   └── 02_model_experiments.ipynb     # Model training and evaluation workflow
│
├── results/                           # Generated outputs
│   └── confusion_matrix_*.png         # Confusion matrices from model training
│
├── templates/                         # Web UI
│   └── index.html                     # Single-page app with textarea, async fetch
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Model Versions

### Baseline (V1)
- **Features**: TF-IDF (10,000 dims) + 7 handcrafted signals (suspicious_word_count, url_count, email_address_count, has_urgency, caps_ratio, text_length, exclamation_count)
- **Model**: Logistic Regression (max_iter=3000, C=1.0, solver="lbfgs")
- **Performance**: 99.46% accuracy, 99.32% precision, 97.90% recall
- **Artifacts**: `models/vectorizer.pkl`, `models/model.pkl`

### Improved (V2)
- **Features**: TF-IDF (10,000 dims) + 9 refined signals (phishing_keyword_count, legitimate_keyword_count, url_count, email_address_count, caps_ratio, text_length, exclamation_count, has_generic_greeting, has_signature)
- **Model**: Logistic Regression with `class_weight='balanced'` to penalize false positives
- **Performance**: 99.59% accuracy, 99.87% precision, 99.30% recall
- **Artifacts**: `models/vectorizer_v2.pkl`, `models/model_v2.pkl`

### Production (Ensemble)
- **Components**: V2 model + rule-based legitimacy override logic
- **Threshold**: 0.70 (vs 0.50 default) for conservative phishing detection
- **Override Logic**: Applies confidence reduction if legitimate signals detected. Override is suppressed when explicit phishing patterns detected.
- **Response Format**: Returns `{label, is_phishing, confidence, phishing_probability, ml_raw_probability, legitimacy_override_applied, reasoning, decision_method}`

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add datasets to `data/`:
   - Place `phishing_legitimate.csv` (text, label columns)
   - Place `spam_assassin.csv` (text, target columns)

## Run Full Pipeline (V2 - Recommended)

```bash
# 1. Normalize and merge datasets
python src/data_preparation.py

# 2. Extract features
python src/feature_engineering_v2.py

# 3. Train models
python src/model_training_v2.py

# 4. (Optional) Analyze threshold tradeoffs
python src/analyze_threshold.py
```

Expected outputs:
- `data/cleaned_emails.csv` (15,796 emails)
- `data/X_features_v2.npz` (feature matrix)
- `data/y_labels_v2.npy` (binary labels)
- `models/vectorizer_v2.pkl` (TF-IDF vectorizer)
- `models/model_v2.pkl` (trained Logistic Regression)
- `results/confusion_matrix_*.png` (if matplotlib/seaborn installed)

## Run Baseline Pipeline (V1)

```bash
python src/data_preparation.py
python src/feature_engineering.py
python src/model_training.py
```

Generates: `models/vectorizer.pkl`, `models/model.pkl`

## Run Web App

```bash
python src/app.py
```

Open http://127.0.0.1:5000 in your browser. The app auto-detects available artifacts and selects the best available predictor (ensemble → v2 → v1).

## Run Tests

```bash
# API validation (requires running app in another terminal)
python tests/test_api.py

# Threshold tuning demonstration
python tests/test_v2_threshold.py
```

## Key Features

- **Auto-detection**: `data_preparation.py` auto-detects text/label columns (supports `label`, `target`, `email_text`, etc.)
- **Label unification**: Maps diverse string labels to binary 0/1 (phishing/spam → 1, legitimate/ham → 0)
- **Lazy loading**: Predictors load artifacts only on first use, preventing import errors
- **Error messages**: API returns clear 503 (missing artifacts) or 500 (runtime error) codes with guidance
- **Ensemble override**: Production system combines ML confidence with rule-based legitimacy signals
- **Explainability**: Response includes detected patterns, reasoning, and decision method
- **Responsive UI**: Single-page app with dynamic result display

## Notes

- False-positive risk identified: legitimate security notifications (password resets) and institutional emails initially flagged as phishing due to dataset bias
- Solution: V2 model with refined features + legitimacy override logic reduces false positives
- Threshold 0.70 balances precision and recall
- `data_preparation.py` auto-detects text/label columns (supports `label`, `target`, etc.)
- Labels are normalized to binary `0/1` even when provided as strings like `spam/ham`
- If model files are missing, API returns a clear error with next-step instructions
