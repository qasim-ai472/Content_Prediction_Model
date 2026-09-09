"""
Task 2 - Content Type Prediction Model
Flask backend: loads the trained model + encoders once at startup,
serves the form, and returns predictions as JSON for the frontend to render.
"""

import json
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE_DIR)

model = joblib.load(os.path.join(BASE_DIR, "best_model.joblib"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.joblib"))
target_encoder = joblib.load(os.path.join(BASE_DIR, "target_encoder.joblib"))

with open(os.path.join(BASE_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_order"]


def safe_encode(column, value):
    """Encode a category the same way training did; fall back to 'Other'
    (or the first known class) if the frontend ever sends something the
    model has never seen."""
    le = encoders[column]
    if value not in le.classes_:
        value = "Other" if "Other" in le.classes_ else le.classes_[0]
    return int(le.transform([value])[0])


@app.route("/")
def home():
    return render_template(
        "index.html",
        genre_options=metadata["genre_options"],
        rating_options=metadata["rating_options"],
        country_options=metadata["country_options"],
        year_min=metadata["release_year_min"],
        year_max=metadata["release_year_max"],
        best_model_name=metadata["best_model_name"],
        results=metadata["results"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    row = {
        "primary_genre": safe_encode("primary_genre", payload.get("genre", "Other")),
        "rating": safe_encode("rating", payload.get("rating", "Not Given")),
        "country_group": safe_encode("country_group", payload.get("country", "Other")),
        "release_year": int(payload.get("release_year", metadata["release_year_max"])),
        "duration_value": float(payload.get("duration_value", 1)),
    }

    X = pd.DataFrame([row])[FEATURE_ORDER]

    prediction = model.predict(X)[0]
    label = target_encoder.inverse_transform([prediction])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = round(float(max(proba)) * 100, 1)

    return jsonify({"prediction": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
