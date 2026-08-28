"""
ReviewSense - Sentiment Analysis Web Application

Serves:
  GET  /                -> Web UI (HTML form)
  POST /api/predict     -> JSON API: {"text": "..."} -> {"label": "...", "confidence": ...}
  GET  /api/health       -> Health check for cloud platform probes
"""
import os
import joblib
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_pipeline.joblib")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Load model once at startup
pipeline = joblib.load(MODEL_PATH)


def predict_sentiment(text: str):
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    label = classes[proba.argmax()]
    confidence = float(proba.max())
    return label, confidence


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    label, confidence = predict_sentiment(text)
    return jsonify({
        "text": text,
        "label": label,
        "confidence": round(confidence, 4),
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
