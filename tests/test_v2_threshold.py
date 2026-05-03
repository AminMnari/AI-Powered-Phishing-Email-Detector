"""Quick test: find optimal threshold for v2 model on problem emails."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictor_v2_improved import predict_email

test_password = """Subject: Your password was changed

Hello,

This is a confirmation that your account password was successfully updated.
If you did not perform this action, please contact support immediately.

Security Team"""

print("Testing password reset email at different thresholds:")
print("=" * 60)

for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    result = predict_email(test_password, threshold=thresh)
    label = "✓ Legitimate" if not result['is_phishing'] else "✗ Phishing"
    print(f"Threshold {thresh}: {label}  (prob={result['phishing_probability']:.1f}%)")

print("\n" + "=" * 60)
print("Issue: V2 model is STILL assigning high probability to legitimate email.")
print("\nRoot cause: The training data likely has these patterns correlating")
print("with phishing more than legitimate emails.")
print("\nSolutions:")
print("1. Use EVEN HIGHER threshold (e.g., 0.85+) - reduces catch rate")
print("2. Add domain/sender validation (separate from ML model)")
print("3. Use ensemble: if signature + professional language → override to Legitimate")
print("4. Collect more diverse legitimate email examples to retrain")
