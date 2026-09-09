"""
Task 2 - Content Type Prediction Model
----------------------------------------
Goal: predict whether a Netflix title is a Movie or a TV Show,
using only the metadata fields available in the catalog (no title text,
no show_id, no director name - those don't generalise to new titles).

Workflow (matches the task brief):
  1. Select relevant dataset features
  2. Encode categorical variables
  3. Train classification models
  4. Evaluate prediction performance
  5. Compare model accuracy

Approach used here:
  - Feature set: primary_genre, rating, country_group, release_year, duration_value
    (duration is stored as "90 min" / "1 Season" in the raw data - we pull out
    just the number, so the model has to learn a genuine pattern from genre,
    rating and country rather than reading the unit label off the field).
  - Rare categories (long-tail countries / genres) are grouped into "Other"
    before encoding, which keeps the encoders small and avoids overfitting
    to categories that appear only a handful of times.
  - Three different classifier families are trained and compared head-to-head:
    Logistic Regression (linear baseline), Random Forest (bagged trees),
    and Gradient Boosting (boosted trees) - so "Step 5: Compare model
    accuracy" is an actual comparison, not a single model reported alone.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset.csv")
MODEL_DIR = BASE_DIR
TOP_N_GENRE = 12
TOP_N_COUNTRY = 8


def extract_duration_number(duration_str):
    """Pull the leading number out of '90 min' / '1 Season' / '9 Seasons'."""
    match = re.match(r"(\d+)", str(duration_str))
    return int(match.group(1)) if match else np.nan


def group_rare_categories(series, top_n):
    """Keep the top_n most frequent categories, bucket everything else as 'Other'."""
    top_values = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_values), other="Other")


def build_features(df):
    df = df.copy()

    # listed_in looks like "Crime TV Shows, International TV Shows, ..."
    # the first genre listed is the most specific one Netflix assigns
    df["primary_genre"] = df["listed_in"].apply(lambda x: str(x).split(",")[0].strip())
    df["primary_genre"] = group_rare_categories(df["primary_genre"], TOP_N_GENRE)

    df["country_group"] = df["country"].fillna("Not Given")
    df["country_group"] = group_rare_categories(df["country_group"], TOP_N_COUNTRY)

    df["duration_value"] = df["duration"].apply(extract_duration_number)
    df["duration_value"] = df["duration_value"].fillna(df["duration_value"].median())

    df["rating"] = df["rating"].fillna("Not Given")

    return df[["primary_genre", "rating", "country_group", "release_year", "duration_value"]]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["type"])

    X_raw = build_features(df)
    y_raw = df["type"]

    # Encode categorical columns; keep the fitted encoders so the frontend
    # can transform new user input the exact same way at prediction time.
    encoders = {}
    X = pd.DataFrame(index=X_raw.index)
    for col in ["primary_genre", "rating", "country_group"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X_raw[col])
        encoders[col] = le

    X["release_year"] = X_raw["release_year"]
    X["duration_value"] = X_raw["duration_value"]

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)  # Movie -> 0, TV Show -> 1 (alphabetical)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    fitted_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        fitted_models[name] = model

        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=target_encoder.classes_))

    print("\n=== Model comparison (Step 5) ===")
    for name, metrics in results.items():
        print(f"{name:20s} accuracy={metrics['accuracy']}  f1={metrics['f1_score']}")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']})")

    # Persist everything the Flask app needs to reproduce the same
    # preprocessing at inference time.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.joblib"))

    metadata = {
        "best_model_name": best_name,
        "feature_order": list(X.columns),
        "genre_options": sorted(X_raw["primary_genre"].unique().tolist()),
        "rating_options": sorted(X_raw["rating"].unique().tolist()),
        "country_options": sorted(X_raw["country_group"].unique().tolist()),
        "release_year_min": int(X_raw["release_year"].min()),
        "release_year_max": int(X_raw["release_year"].max()),
        "results": results,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model, encoders and metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
