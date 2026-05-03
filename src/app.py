from flask import Flask, render_template, request, jsonify

try:
    from predictor_ensemble import predict_email_ensemble as predict_email
except ImportError:
    from predictor import predict_email

app = Flask(__name__, template_folder="../templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json() or {}
        email_text = data.get("email_text", "")
    else:
        email_text = request.form.get("email_text", "")

    if not email_text.strip():
        return jsonify({"error": "No email text provided"}), 400

    try:
        result = predict_email(email_text)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "fix": "Run python src/feature_engineering_v2.py and python src/model_training_v2.py"}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
