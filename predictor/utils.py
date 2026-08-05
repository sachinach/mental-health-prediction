"""
Utility layer for the predictor app.

Keeps everything model/IO related out of views.py: loading the trained
artifacts, transforming raw form input into the exact feature vector the
model was trained on, running inference, and appending results to the
CSV-backed prediction history.
"""

import csv
import json
import logging
import pickle
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from django.conf import settings

logger = logging.getLogger("predictor")


class ModelLoadError(Exception):
    """Raised when the trained model or preprocessing bundle can't be loaded."""


class PredictionError(Exception):
    """Raised when a prediction can't be produced from the given input."""


@lru_cache(maxsize=1)
def load_model():
    """Load the trained classifier and preprocessing bundle from disk.

    Cached with lru_cache so the pickle files are only read once per
    process, not on every request.
    """
    try:
        with open(settings.MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(settings.PREPROCESSING_PATH, "rb") as f:
            preprocessing = pickle.load(f)
    except FileNotFoundError as exc:
        logger.error("Model artifacts not found: %s", exc)
        raise ModelLoadError(
            "Trained model files were not found. Make sure model.pkl and "
            "preprocessing.pkl exist inside the model/ directory."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load model artifacts: %s", exc)
        raise ModelLoadError(f"Could not load model artifacts: {exc}") from exc

    return model, preprocessing


def get_form_choices():
    """Expose the categorical options the model was trained on, so the
    prediction form always stays in sync with preprocessing.pkl."""
    _, preprocessing = load_model()
    return {
        "genders": preprocessing["gender_classes"],
        "occupations": preprocessing["occupation_classes"],
    }


def preprocess_input(raw_data: dict) -> pd.DataFrame:
    """Transform a dict of raw form values into the exact feature matrix
    the trained model expects (same cleaning / engineering / encoding /
    scaling steps used during training).
    """
    _, preprocessing = load_model()

    try:
        row = {
            "age": float(raw_data["age"]),
            "sleep_hours": float(raw_data["sleep_hours"]),
            "sleep_quality": float(raw_data["sleep_quality"]),
            "social_media_hours": float(raw_data["social_media_hours"]),
            "academic_work_pressure": float(raw_data["academic_work_pressure"]),
            "physical_activity_days": float(raw_data["physical_activity_days"]),
            "stress_level": float(raw_data["stress_level"]),
            "anxiety_score": float(raw_data["anxiety_score"]),
            "depression_score": float(raw_data["depression_score"]),
            "work_life_balance": float(raw_data["work_life_balance"]),
            "mood_score": float(raw_data["mood_score"]),
            "concentration_level": float(raw_data["concentration_level"]),
            "social_support": float(raw_data["social_support"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionError(f"Invalid or missing input value: {exc}") from exc

    gender = raw_data.get("gender")
    occupation = raw_data.get("occupation")
    if gender not in preprocessing["gender_classes"]:
        raise PredictionError(f"Unrecognised gender option: {gender}")
    if occupation not in preprocessing["occupation_classes"]:
        raise PredictionError(f"Unrecognised occupation option: {occupation}")

    # Feature engineering - identical formulas used at training time
    row["Wellbeing_Index"] = (
        row["sleep_hours"] + row["physical_activity_days"] + row["mood_score"]
        - row["social_media_hours"] / 2
    )
    row["Pressure_Index"] = row["academic_work_pressure"] * (10 - row["work_life_balance"]) / 10

    # Encoding - label encode gender using the fitted encoder
    le_gender = preprocessing["le_gender"]
    row["Gender_Enc"] = int(le_gender.transform([gender])[0])

    # One-hot encode occupation exactly as pd.get_dummies(drop_first=True) did
    for col in preprocessing["occ_cols"]:
        occ_value = col.replace("Occ_", "")
        row[col] = 1 if occupation == occ_value else 0

    df = pd.DataFrame([row])

    # Select and order columns exactly as the trained model expects,
    # filling any engineered dummy columns that don't apply with 0
    feature_names = preprocessing["feature_names"]
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale the same numeric columns using the fitted scaler
    scale_cols = preprocessing["scale_cols"]
    df[scale_cols] = preprocessing["scaler"].transform(df[scale_cols])

    return df


def predict_risk(raw_data: dict) -> dict:
    """Run the full inference pipeline and return a structured result."""
    model, _ = load_model()
    features = preprocess_input(raw_data)

    # The model was fit on a plain ndarray (post-SMOTE balancing), so
    # predict on .values to avoid sklearn's "fitted without feature
    # names" warning while keeping preprocess_input's DataFrame output
    # for readability/debugging.
    prediction = int(model.predict(features.values)[0])
    probabilities = model.predict_proba(features.values)[0]

    label = "High Risk" if prediction == 1 else "Low Risk"
    confidence = float(probabilities[prediction])

    recommendations = _get_recommendations(prediction, raw_data)

    return {
        "prediction": prediction,
        "label": label,
        "confidence": round(confidence * 100, 2),
        "probability_high_risk": round(float(probabilities[1]) * 100, 2),
        "probability_low_risk": round(float(probabilities[0]) * 100, 2),
        "recommendations": recommendations,
    }


def _get_recommendations(prediction: int, raw_data: dict) -> list:
    """Return a short, targeted list of suggestions based on the outcome
    and the weakest-looking inputs supplied."""
    recs = []

    if prediction == 1:
        recs.append("Consider speaking with a qualified mental health professional or counselor.")
    else:
        recs.append("Keep up the habits that are working - regular check-ins help sustain them.")

    try:
        if float(raw_data.get("sleep_hours", 8)) < 6:
            recs.append("Aim for 7-9 hours of sleep a night; sleep debt compounds stress over time.")
        if float(raw_data.get("physical_activity_days", 3)) < 2:
            recs.append("Build in light physical activity a few times a week - even short walks help.")
        if float(raw_data.get("stress_level", 5)) >= 7:
            recs.append("Practice a daily stress-reduction routine such as meditation or deep breathing.")
        if float(raw_data.get("social_support", 5)) <= 3:
            recs.append("Lean on your support network more - a short check-in with someone you trust helps.")
        if float(raw_data.get("social_media_hours", 2)) > 5:
            recs.append("Try setting boundaries around social media / screen time, especially before bed.")
        if float(raw_data.get("work_life_balance", 5)) <= 3:
            recs.append("Look for small ways to protect personal time from academic or work demands.")
    except (TypeError, ValueError):
        pass

    return recs[:5]


def save_prediction(raw_data: dict, result: dict) -> None:
    """Append a prediction record to the CSV-backed history file."""
    settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = settings.PREDICTION_HISTORY_PATH.exists()

    try:
        with open(settings.PREDICTION_HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Age", "Gender", "Occupation", "Prediction", "Probability"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                raw_data.get("age"),
                raw_data.get("gender"),
                raw_data.get("occupation"),
                result["label"],
                f"{result['probability_high_risk']}%",
            ])
    except OSError as exc:
        logger.error("Failed to write prediction history: %s", exc)
        raise PredictionError(f"Could not save prediction to history: {exc}") from exc


def read_history() -> list:
    """Read the full prediction history back out, most recent first."""
    if not settings.PREDICTION_HISTORY_PATH.exists():
        return []

    try:
        with open(settings.PREDICTION_HISTORY_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    except OSError as exc:
        logger.error("Failed to read prediction history: %s", exc)
        return []

    records.reverse()
    return records


def get_model_metrics() -> dict:
    """Expose the held-out test metrics captured at training time, for
    display on the About page."""
    _, preprocessing = load_model()
    return {
        "model_name": preprocessing.get("final_model_name", "Random Forest"),
        **preprocessing.get("test_metrics", {}),
    }


@lru_cache(maxsize=1)
def get_analytics() -> dict:
    """Load the training-time analytics bundle (per-algorithm metrics,
    including F1 score, plus references to the saved comparison plots)
    generated by train_pipeline.py."""
    analytics_path = settings.MODEL_DIR / "analytics.json"
    if not analytics_path.exists():
        raise ModelLoadError(
            "model/analytics.json was not found. Run `python train_pipeline.py` "
            "from the project root to regenerate it."
        )
    with open(analytics_path, "r", encoding="utf-8") as f:
        return json.load(f)
