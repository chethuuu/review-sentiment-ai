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
    idx = proba.argmax()
    label = classes[idx]
    confidence = float(proba[idx])

    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    x = vectorizer.transform([text])
    # coef_ points toward classes_[1]; flip the sign when the predicted
    # label is classes_[0] so "contribution" always means "pushed toward
    # the predicted label".
    sign = 1 if label == classes[1] else -1

    contributions = [
        (feature_names[i], sign * coefs[i] * x[0, i])
        for i in x.nonzero()[1]
    ]
    contributions.sort(key=lambda pair: pair[1], reverse=True)
    key_phrases = [word for word, score in contributions[:5] if score > 0]

    return label, confidence, key_phrases


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400

    label, confidence, key_phrases = predict_sentiment(text)
    return jsonify({
        "text": text,
        "label": label,
        "confidence": round(confidence, 4),
        "key_phrases": key_phrases,
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
