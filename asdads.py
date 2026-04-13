import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import json
import sys
import os

model = lgb.Booster(model_file="C:/Users/Lenovo/Downloads/blood_disorder_lgbm.txt")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

def predict_single(input_dict, threshold=0.20):
    row = {col: np.nan for col in feature_cols}
    for k, v in input_dict.items():
        if k in row:
            row[k] = v
    for col in [c for c in feature_cols if not c.endswith("__present")]:
        pres_col = f"{col}__present"
        if pres_col in feature_cols:
            row[pres_col] = 0 if pd.isna(row[col]) else 1
    df_input = pd.DataFrame([row])[feature_cols]
    probs = model.predict(df_input)[0]
    pred_idx = np.argmax(probs)
    pred_class = le.classes_[pred_idx]
    confidence = probs[pred_idx]
    return {
        "predicted_class":        pred_class,
        "confidence":             round(float(confidence), 4),
        "all_probabilities":      {le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "low_confidence_warning": confidence < threshold,
    }

def run_from_json(input_path, output_path=None):
    with open(input_path, "r") as f:
        input_dict = json.load(f)
    result = predict_single(input_dict)
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Predicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence']:.2%}")
    print(f"Low confidence? : {result['low_confidence_warning']}")
    print(f"\nAll probabilities:")
    for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
        print(f"  {cls:25s}: {prob:.4f}")
    print(f"\nResult saved to : {output_path}")

if len(sys.argv) < 2:
    print("Usage: python inference.py input.json [output.json]")
    print("Or set USE_JSON = False and use the hardcoded input_dict.")
    sys.exit(1)
input_path  = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else None
run_from_json(input_path, output_path)
