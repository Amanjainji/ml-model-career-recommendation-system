import os
import traceback
import numpy as np
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Path to model (default = model/career_model.joblib)
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "model/career_model.joblib")

# Load model
if not os.path.exists(MODEL_LOCAL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_LOCAL_PATH}")

model = joblib.load(MODEL_LOCAL_PATH)

# Features and career labels (from training info)
FEATURE_NAMES = [
    "Math_Score","Science_Score","English_Score","Social_Score",
    "Logical_Reasoning","Verbal_Ability","Numerical_Ability",
    "Spatial_Reasoning","Abstract_Reasoning",
    "Openness","Conscientiousness","Extraversion",
    "Agreeableness","Neuroticism",
    "Tech_Interest","Creative_Interest","People_Interest",
    "Analytical_Interest","Leadership_Interest"
]

CAREER_CLASSES = [
    "Arts & Humanities",
    "Business",
    "Computer Science",
    "Engineering",
    "Law",
    "Medical",
    "Research",
    "Teaching"
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate all features
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Convert to array
        features = [data[f] for f in FEATURE_NAMES]
        features = np.array(features, dtype=float).reshape(1, -1)

        # Predict
        pred = model.predict(features)[0]

        # If classifier outputs index
        if isinstance(pred, (np.integer, int)):
            career = CAREER_CLASSES[pred]
        else:
            career = str(pred)

        result = {"recommended_career": career}

        # Confidence score (optional)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features).max().item()
            result["confidence"] = round(float(prob), 4)

        return jsonify(result)

    except Exception as e:
        print("ERROR in /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
