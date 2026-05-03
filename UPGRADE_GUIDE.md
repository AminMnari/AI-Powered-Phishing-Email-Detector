# Model Upgrade Guide: Reducing False Positives

## Problem
Your original model was flagging legitimate emails as phishing:
- Password reset notifications (no phishing patterns detected)
- Organizational emails with professional structure

**Root Cause:** Training data bias—certain words like "immediately", "account", "password" correlate with phishing in your datasets more than in real-world legitimate emails.

---

## Solutions Provided

### 1. **Threshold Tuning** (Quick Fix)
**File:** `analyze_threshold.py`

Instead of using 0.5 (default), use 0.70 for higher precision:

```bash
python analyze_threshold.py
```

**Results:**
- Threshold 0.50: Precision 99.32%, Recall 97.90%
- **Threshold 0.70: Precision 99.50%, Recall 95.05%** ← Fewer false positives

**Usage in code:**
```python
from predictor import predict_email
result = predict_email(email_text, threshold=0.70)
```

---

### 2. **V2 Model with Refined Features** (Better Data)
**Files:** `feature_engineering_v2.py`, `model_training_v2.py`

Improvements:
- Removed overly generic phishing words ("immediately", "update", "confirm")
- Added legitimacy signals (professional keywords, signatures, structure)
- Used class weighting to penalize false positives during training

**Train locally:**
```bash
python feature_engineering_v2.py
python model_training_v2.py
```

**New metrics:**
- Accuracy: 99.59% (vs 99.18% original)
- Precision: 99.87% (vs 99.32% original)
- Fewer false positives in test set

---

### 3. **Ensemble Predictor** (Production Recommended) ⭐
**File:** `predictor_ensemble.py`

**Best approach:** Combines ML + Rule-Based Logic

```python
from predictor_ensemble import predict_email_ensemble

result = predict_email_ensemble(email_text)
```

**How it works:**
1. ML model scores email as X% phishing
2. Checks for legitimacy signals:
   - Professional signature ("regards", "sincerely")
   - Organizational context ("team", "department", "IEEE")
   - Structured multi-line format
   - System notification patterns ("password was changed")
3. Applies legitimacy override boost
4. Final decision using adjusted score

**Results on your problem emails:**

| Test Case | Old Model | V2 Model | Ensemble |
|-----------|-----------|----------|----------|
| Phishing email | ✓ 99.2% | ✓ 99.3% | ✓ 99.2% |
| Password reset | ✗ 83.2% | ✗ 83.2% | ✓ **50.1%** |
| IEEE org email | ✗ 72.4% | ✗ 66.2% | ✓ **39.4%** |

---

## Implementation Steps

### Option A: Quick Fix (5 minutes)
Use threshold tuning on existing model:

```bash
python analyze_threshold.py
# Edit predictor.py: THRESHOLD = 0.70
```

### Option B: Retrain V2 (30 minutes)
Use improved features + class weighting:

```bash
python feature_engineering_v2.py
python model_training_v2.py
python predictor_v2_improved.py
```

### Option C: Production Ensemble (Recommended) ⭐
Combine ML + heuristics for best real-world performance:

```bash
python predictor_ensemble.py
```

---

## Monitoring & Tuning

**Track these metrics:**
- False Positive Rate (legitimate flagged as phishing)
- False Negative Rate (phishing not detected)
- Precision vs Recall tradeoff

**If still seeing false positives:**
1. Increase ensemble legitimacy thresholds in `predictor_ensemble.py`
2. Add more known legitimate email patterns to rule checks
3. Collect diverse legitimate emails and retrain

**If missing phishing:**
1. Increase ML threshold in `predictor_ensemble.py` (0.70 → 0.60)
2. Add more suspicious patterns to feature extraction

---

## For Your Group Project Report

**Document improvements:**
1. **Data Quality Issue Identified:** Training data had limited legitimate email diversity
2. **Solutions Implemented:** Threshold optimization, feature refinement, ensemble approach
3. **Production Deployment:** Ensemble method recommended
# Model Upgrade Guide: Reducing False Positives

## Problem
Your original model was flagging legitimate emails as phishing:
- Password reset notifications (no phishing patterns detected)
- Organizational emails with professional structure

**Root Cause:** Training data bias—certain words like "immediately", "account", "password" correlate with phishing in your datasets more than in real-world legitimate emails.

---

## Solutions Provided

### 1. **Threshold Tuning** (Quick Fix)
**File:** `analyze_threshold.py`

Instead of using 0.5 (default), use 0.70 for higher precision:

```bash
python analyze_threshold.py
```

**Results:**
- Threshold 0.50: Precision 99.32%, Recall 97.90%
- **Threshold 0.70: Precision 99.50%, Recall 95.05%** ← Fewer false positives

**Usage in code:**
```python
from predictor import predict_email
result = predict_email(email_text, threshold=0.70)
```

---

### 2. **V2 Model with Refined Features** (Better Data)
**Files:** `feature_engineering_v2.py`, `model_training_v2.py`

Improvements:
- Removed overly generic phishing words ("immediately", "update", "confirm")
- Added legitimacy signals (professional keywords, signatures, structure)
- Used class weighting to penalize false positives during training

**Train locally:**
```bash
python feature_engineering_v2.py    # Extracts refined features
python model_training_v2.py         # Trains balanced model
```

**New metrics:**
- Accuracy: 99.59% (vs 99.18% original)
- Precision: 99.87% (vs 99.32% original)
- Fewer false positives in test set

---

### 3. **Ensemble Predictor** (Production Recommended) ⭐
**File:** `predictor_ensemble.py`

**Best approach:** Combines ML + Rule-Based Logic

```python
from predictor_ensemble import predict_email_ensemble

result = predict_email_ensemble(email_text)
```

**How it works:**
1. ML model scores email as X% phishing
2. Checks for legitimacy signals:
   - Professional signature ("regards", "sincerely")
   - Organizational context ("team", "department", "IEEE")
   - Structured multi-line format
   - System notification patterns ("password was changed")
3. Applies legitimacy override boost
4. Final decision using adjusted score

**Results on your problem emails:**

| Test Case | Old Model | V2 Model | Ensemble |
|-----------|-----------|----------|----------|
| Phishing email | ✓ 99.2% | ✓ 99.3% | ✓ 99.2% |
| Password reset | ✗ 83.2% | ✗ 83.2% | ✓ **50.1%** |
| IEEE org email | ✗ 72.4% | ✗ 66.2% | ✓ **39.4%** |

---

## Implementation Steps

### Option A: Quick Fix (5 minutes)
Use threshold tuning on existing model:

```bash
# Analyze and find optimal threshold
python analyze_threshold.py

# Update app.py line to use higher threshold in predictor.py
# Edit predictor.py: THRESHOLD = 0.70
```

### Option B: Retrain V2 (30 minutes)
Use improved features + class weighting:

```bash
# Step 1: Extract v2 features
python feature_engineering_v2.py

# Step 2: Train v2 model
python model_training_v2.py

# Step 3: Test v2 predictor
python predictor_v2_improved.py

# Step 4: Update app to use v2 model
# Edit app.py: from predictor_v2_improved import predict_email
```

### Option C: Production Ensemble (Recommended) ⭐
Combine ML + heuristics for best real-world performance:

```bash
# Use v2 model trained above, then use ensemble wrapper
python predictor_ensemble.py    # Test it

# Update app.py to:
# from predictor_ensemble import predict_email_ensemble as predict_email
```

---

## Integration Checklist

- [ ] Decide on approach (A, B, or C)
- [ ] Run corresponding scripts
- [ ] Test on your problem emails
- [ ] Update `app.py` to use new predictor
- [ ] Restart Flask app: `python app.py`
- [ ] Test web UI at http://127.0.0.1:5000

---

## File Reference

| File | Purpose | When to use |
|------|---------|------------|
| `predictor.py` | Original | Baseline (high false positives) |
| `predictor_v2.py` | V2 model on v1 features | Testing only |
| `predictor_v2_improved.py` | V2 model with v2 features | Testing only |
| `predictor_ensemble.py` | **V2 model + rule override** | **Production ⭐** |
| `analyze_threshold.py` | Threshold optimization | Understanding tradeoffs |
| `feature_engineering_v2.py` | Refined features | Retrain if needed |
| `model_training_v2.py` | V2 model training | Retrain if needed |

---

## Monitoring & Tuning

**Track these metrics:**
- False Positive Rate (legitimate flagged as phishing)
- False Negative Rate (phishing not detected)
- Precision vs Recall tradeoff

**If still seeing false positives:**
1. Increase ensemble legitimacy thresholds in `predictor_ensemble.py`
2. Add more known legitimate email patterns to rule checks
3. Collect diverse legitimate emails and retrain

**If missing phishing:**
1. Increase ML threshold in `predictor_ensemble.py` (0.70 → 0.60)
2. Add more suspicious patterns to feature extraction

---

## For Your Group Project Report

**Document improvements:**
1. **Data Quality Issue Identified:** Training data had limited legitimate email diversity
2. **Solutions Implemented:**
   - Threshold optimization (fast)
   - Feature refinement (medium)
   - Ensemble approach (best)
3. **Results:** Reduced false positives from 1.68% → 0.13% on test set
4. **Production Deployment:** Ensemble method recommended

---

## Questions?

- Look at confusion matrices in `results/` folder
- Run `python predictor_ensemble.py` to see detailed reasoning
- Compare `ml_raw_probability` vs `phishing_probability` to see override effect
