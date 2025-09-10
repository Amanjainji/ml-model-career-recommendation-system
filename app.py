import os
import pickle
import traceback
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Paths (Render will read from repo or environment)
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "model/trained_career_model.pkl")

# Features and career labels (fixed from your training info)
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

# Load model at startup
if not os.path.exists(MODEL_LOCAL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_LOCAL_PATH}")

with open(MODEL_LOCAL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate all features are present
        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Extract features in correct order
        features = [data[f] for f in FEATURE_NAMES]
        features = np.array(features, dtype=float).reshape(1, -1)

        # Predict
        pred_index = model.predict(features)[0]
        # If GradientBoostingClassifier trained on encoded labels, pred_index is already int index
        if isinstance(pred_index, (np.integer, int)):
            career = CAREER_CLASSES[pred_index]
        else:
            # In case model.predict outputs label string directly
            career = str(pred_index)

        result = {"recommended_career": career}

        # Optionally add confidence score
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features).max().item()
            result["confidence"] = round(float(prob), 4)

        return jsonify(result)

    except Exception as e:
        print("ERROR in /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500
