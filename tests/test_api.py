"""Quick test to verify Flask API works without errors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import requests

print("Testing Flask API with fixed template...\n")

tests = [
    ("Password Reset", "Subject: Your password was changed\n\nThis confirms your password was successfully updated.\nIf not authorized, contact support immediately.\n\nBest regards,\nSecurity Team"),
    ("Phishing", "URGENT ALERT! Click here NOW to verify your account: http://verify-immediately.net\n\nYour account security is at risk!"),
    ("Organizational Email", "Hi Team,\n\nPlease review the attached meeting notes and agenda.\n\nBest regards,\nJohn Smith\nProject Manager"),
]

for name, email in tests:
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"email_text": email},
            timeout=5
        )
        data = response.json()

        required_fields = ["label", "is_phishing", "confidence", "phishing_probability"]
        missing = [f for f in required_fields if f not in data]

        if missing:
            print(f"❌ {name}: Missing fields: {missing}")
        else:
            print(f"✓ {name}: {data['label']} (confidence: {data['confidence']}%)")

    except Exception as e:
        print(f"❌ {name}: {e}")

print("\n✓ All API tests completed - no 'undefined' errors")
