from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from google import genai
from google.genai import types
import json

app = Flask(__name__)
CORS(app)

model = lgb.Booster(model_file="blood_disorder_lgbm.txt")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

RAW_FEATURES = [
    "age", "gender",
    "fasting_glucose", "HbA1c", "HOMA_IR",
    "hemoglobin", "RBC", "hematocrit", "MCV", "MCH", "MCHC", "RDW", "platelets",
    "neutrophils_pct", "lymphocytes_pct", "monocytes_pct", "eosinophils_pct", "basophils_pct",
    "TSH", "Free_T3", "Free_T4", "anti_TPO",
    "creatinine", "urea_BUN", "eGFR", "uric_acid",
    "ALT", "AST", "ALP", "GGT", "bilirubin_total", "bilirubin_direct", "albumin", "total_protein",
    "total_cholesterol", "LDL", "HDL", "VLDL", "triglycerides", "non_HDL",
    "WBC", "CRP", "ESR", "procalcitonin",
    "Vitamin_D", "B12", "folate", "iron", "ferritin", "TIBC", "transferrin_sat", "zinc",
    "sodium", "potassium", "chloride", "bicarbonate", "calcium", "magnesium", "phosphorus", "anion_gap",
    "cortisol", "testosterone", "estrogen", "FSH", "LH", "prolactin", "DHEA_S",
    "PT", "INR", "aPTT", "fibrinogen", "D_dimer", "thrombin_time",
    "RBS", "PPG"
]

EXTRACTION_PROMPT = f"""
You are a medical data extractor. From the given medical report PDF, extract the following lab values and return ONLY a valid JSON object with these exact keys.

Rules:
- All values must be numeric.
- "gender": 0 for female, 1 for male.
- "age": number (e.g. 34).
- If a value is not present in the report, use null.

Keys: {json.dumps(RAW_FEATURES)}

Return ONLY the JSON. No explanation. No markdown. No code fences.
"""

GEMINI_API_KEY = "AIzaSyDbBUcV3M86d9vhHivdzAioNlvP1AQVlGY"


def build_input_vector(extracted: dict) -> pd.DataFrame:
    row = {col: np.nan for col in feature_cols}
    for feat in RAW_FEATURES:
        if feat in row and extracted.get(feat) is not None:
            row[feat] = float(extracted[feat])
    for feat in RAW_FEATURES:
        pres_col = f"{feat}__present"
        if pres_col in row:
            val = row.get(feat)
            try:
                row[pres_col] = 0 if (val is None or np.isnan(float(val))) else 1
            except (TypeError, ValueError):
                row[pres_col] = 0
    return pd.DataFrame([row])[feature_cols]


def generate_summary(client, prediction: str, confidence: float, extracted: dict) -> dict:
    """
    Second Gemini call: generates a clinical conclusion and next steps
    based on the model prediction and extracted lab values.
    Returns {"conclusion": "...", "next_steps": ["...", "..."]}
    """
    non_null = {k: v for k, v in extracted.items() if v is not None}
    values_text = ", ".join(f"{k}={v}" for k, v in non_null.items())

    summary_prompt = f"""
You are a clinical assistant helping a doctor interpret lab results.

The AI model has classified this patient's blood report as: "{prediction}" with {confidence*100:.1f}% confidence.

Extracted lab values: {values_text}

Based on this, provide:
1. A concise clinical conclusion (2-3 sentences max) explaining what the findings suggest.
2. A list of 4-6 recommended next steps for the doctor (tests, referrals, lifestyle changes, monitoring).

Respond ONLY with a valid JSON object in this exact format:
{{
  "conclusion": "...",
  "next_steps": ["...", "...", "..."]
}}

No markdown. No explanation. Just the JSON.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[summary_prompt]
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "conclusion": f"Summary generation failed: {str(e)}",
            "next_steps": []
        }


@app.route("/predict", methods=["POST"])
def predict():
    file_key = "pdf" if "pdf" in request.files else "file"
    if file_key not in request.files:
        return jsonify({"error": "No PDF uploaded. Send file with key 'pdf' or 'file'"}), 400

    pdf_bytes = request.files[file_key].read()

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                EXTRACTION_PROMPT
            ]
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        extracted = json.loads(raw)
    except Exception as e:
        return jsonify({"error": f"Gemini extraction failed: {str(e)}"}), 500

    try:
        df_input = build_input_vector(extracted)
    except Exception as e:
        return jsonify({"error": f"Feature vector build failed: {str(e)}"}), 500

    try:
        probs     = model.predict(df_input)[0]
        pred_idx  = int(np.argmax(probs))
        pred_class= le.classes_[pred_idx]
        confidence= round(float(probs[pred_idx]), 4)
        all_probs = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)}
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500

    summary = generate_summary(client, pred_class, confidence, extracted)

    return jsonify({
        "prediction":            pred_class,
        "confidence":            confidence,
        "all_probabilities":     all_probs,
        "extracted_features":    extracted,
        "summary":               summary,
        "low_confidence_warning":confidence < 0.20
    })


if __name__ == "__main__":
    app.run(debug=True)