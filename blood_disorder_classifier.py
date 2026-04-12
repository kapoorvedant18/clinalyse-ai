"""
Blood Disorder Classification Pipeline
- LightGBM multiclass classifier
- Structured panel dropout for training robustness
- Binary presence indicators for all features
- 70/15/15 train/val/test split
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

BASE_PATH = "C:/Users/Lenovo/Downloads/"

FILES = {
    "Diabetes":              "C:/Users/Lenovo/Downloads/diabetes_csv_updated.csv",
    "Normal":                "C:/Users/Lenovo/Downloads/All_Normal_csv_updated.csv",
    "Clotting":              "C:/Users/Lenovo/Downloads/Clotting_1000samples_final_csv_updated.csv",
    "Hormonal_Endocrine":    "C:/Users/Lenovo/Downloads/Hormonal_Endocrine_rightmost_only_csv_updated.csv",
    "Infection_Inflammation":"C:/Users/Lenovo/Downloads/Infection_Inflammation_1000samples_final_csv_updated.csv",
    "Lipid_CV":              "C:/Users/Lenovo/Downloads/Lipid_CV_1000samples_final_csv_updated.csv",
    "Liver":                 "C:/Users/Lenovo/Downloads/Liver_1500samples_csv_updated.csv",
    "Nutritional":           "C:/Users/Lenovo/Downloads/Nutritional_1500samples_final_csv_updated.csv",
    "Renal":                 "C:/Users/Lenovo/Downloads/Renal_1000samples_final_csv_updated.csv",
    "Electrolyte":           "C:/Users/Lenovo/Downloads/Electrolyte_1500samples_final_csv_updated.csv",
    "Hematological":         "C:/Users/Lenovo/Downloads/Hematological_2000_formatted_csv_updated.csv",
    "Thyroid":               "C:/Users/Lenovo/Downloads/Thyroid_1000samples_final_csv__1__updated.csv",
}

# Columns to always drop (non-feature)
DROP_COLS = ["sample_id", "disease_focus"]

# ─────────────────────────────────────────────
# 2. PANEL DEFINITIONS
#    Keys match FILES keys exactly
# ─────────────────────────────────────────────

PANELS = {
    "Diabetes": [
        "fasting_glucose", "HbA1c", "HOMA_IR", "RBS", "PPG"
    ],
    "Hematological": [
        "hemoglobin", "RBC", "hematocrit", "MCV", "MCH", "MCHC", "RDW",
        "platelets", "neutrophils_pct", "lymphocytes_pct", "monocytes_pct",
        "eosinophils_pct", "basophils_pct"
    ],
    "Thyroid": [
        "TSH", "Free_T3", "Free_T4", "anti_TPO"
    ],
    "Renal": [
        "creatinine", "urea_BUN", "eGFR", "uric_acid"
    ],
    "Liver": [
        "ALT", "AST", "ALP", "GGT", "bilirubin_total", "bilirubin_direct",
        "albumin", "total_protein"
    ],
    "Lipid_CV": [
        "total_cholesterol", "LDL", "HDL", "VLDL", "triglycerides", "non_HDL"
    ],
    "Infection_Inflammation": [
        "WBC", "CRP", "ESR", "procalcitonin"
    ],
    "Nutritional": [
        "Vitamin_D", "B12", "folate", "iron", "ferritin", "TIBC",
        "transferrin_sat", "zinc"
    ],
    "Electrolyte": [
        "sodium", "potassium", "chloride", "bicarbonate", "calcium",
        "magnesium", "phosphorus", "anion_gap"
    ],
    "Hormonal_Endocrine": [
        "cortisol", "testosterone", "estrogen", "FSH", "LH",
        "prolactin", "DHEA_S"
    ],
    "Clotting": [
        "PT", "INR", "aPTT", "fibrinogen", "D_dimer", "thrombin_time"
    ],
}

# All panels as a flat list of (panel_name, [features]) — used for dropout
ALL_PANELS = list(PANELS.items())

PANEL_DROPOUT_PROB = 0.75       # probability of dropping a non-defining panel
WITHIN_PANEL_DROPOUT = 0.30     # individual feature dropout within defining panel

# ─────────────────────────────────────────────
# 3. LOAD & FILTER
# ─────────────────────────────────────────────

def load_and_filter(class_name, filename):
    df = pd.read_csv(os.path.join(BASE_PATH, filename))
    status_col = df.columns[-1]  # always last column

    if class_name == "Normal":
        # Keep only rows where status contains 'normal' (case-insensitive)
        mask = df[status_col].str.contains("normal", case=False, na=False)
    else:
        # Keep only rows where status contains 'Abnormal' (case-insensitive)
        mask = df[status_col].str.contains("abnormal", case=False, na=False)

    df = df[mask].copy()

    # Drop non-feature columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    cols_to_drop.append(status_col)  # drop status column
    df = df.drop(columns=cols_to_drop)

    df["label"] = class_name
    return df


def load_all():
    dfs = []
    for class_name, filename in FILES.items():
        df = load_and_filter(class_name, filename)
        print(f"  {class_name:25s}: {len(df)} rows after filtering")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ─────────────────────────────────────────────
# 4. FEATURE COLUMNS
# ─────────────────────────────────────────────

def get_feature_cols(df):
    return [c for c in df.columns if c != "label"]

# ─────────────────────────────────────────────
# 5. PRESENCE INDICATORS
#    Added BEFORE dropout so they reflect original data presence
#    (i.e. whether the value existed in the source data)
# ─────────────────────────────────────────────

def add_presence_indicators(df, feature_cols):
    for col in feature_cols:
        df[f"{col}__present"] = df[col].notna().astype(np.int8)
    return df

# ─────────────────────────────────────────────
# 6. STRUCTURED PANEL DROPOUT (training only)
# ─────────────────────────────────────────────

def apply_panel_dropout(df, feature_cols, rng=None):
    """
    For each row:
      - Identify the defining panel for that row's class
      - Never fully drop the defining panel; apply 30% individual dropout within it
      - For all other panels: drop the entire panel with 75% probability
      - Normal class: all panels subject to 75% hard dropout + 30% within kept panels
    Returns a copy with NaNs injected.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df_out = df.copy()
    feature_set = set(feature_cols)

    for idx in df_out.index:
        cls = df_out.at[idx, "label"]
        defining_panel_features = set(PANELS.get(cls, []))

        for panel_name, panel_features in ALL_PANELS:
            # Only process features that actually exist in our dataframe
            present_features = [f for f in panel_features if f in feature_set]
            if not present_features:
                continue

            is_defining = (panel_name == cls)

            if cls == "Normal":
                # All panels treated symmetrically for Normal
                if rng.random() < PANEL_DROPOUT_PROB:
                    df_out.loc[idx, present_features] = np.nan
                else:
                    # Within kept panel, 30% individual dropout
                    for feat in present_features:
                        if rng.random() < WITHIN_PANEL_DROPOUT:
                            df_out.at[idx, feat] = np.nan

            elif is_defining:
                # Never fully drop — apply 30% individual feature dropout
                for feat in present_features:
                    if rng.random() < WITHIN_PANEL_DROPOUT:
                        df_out.at[idx, feat] = np.nan

            else:
                # Non-defining panel: 75% hard drop
                if rng.random() < PANEL_DROPOUT_PROB:
                    df_out.loc[idx, present_features] = np.nan

    return df_out

# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Loading and filtering data...")
    print("=" * 60)
    df = load_all()
    print(f"\nTotal rows: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}\n")

    feature_cols = get_feature_cols(df)
    print(f"Feature columns: {len(feature_cols)}")

    # -- Encode gender globally before split
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).astype("float32")

    # ── Add presence indicators (on original data, before any dropout)
    print("\nAdding presence indicators...")
    df = add_presence_indicators(df, feature_cols)

    # ── Encode labels
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    print(f"Classes: {list(le.classes_)}")

    # ── Train/Val/Test split (70/15/15), stratified
    print("\nSplitting data 70/15/15...")
    X = df.drop(columns=["label", "label_enc"])
    y = df["label_enc"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Attach label back to train for dropout (needed to know defining panel per row)
    train_labels = df.loc[X_train.index, "label"]
    X_train = X_train.copy()
    X_train["label"] = train_labels.values

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Apply structured panel dropout to training set only
    print("\nApplying structured panel dropout to training set...")
    rng = np.random.default_rng(42)
    X_train_dropped = apply_panel_dropout(X_train, feature_cols, rng=rng)
    X_train_final = X_train_dropped.drop(columns=["label"])

    # ── Update presence indicators AFTER dropout
    #    (so model sees dropout-aware presence signals during training)
    print("Updating presence indicators post-dropout for training set...")
    for col in feature_cols:
        pres_col = f"{col}__present"
        if pres_col in X_train_final.columns:
            X_train_final[pres_col] = X_train_final[col].notna().astype(np.int8)

    # Val/test: presence indicators already set from original data — no changes needed

    # ── LightGBM training
    print("\nTraining LightGBM...")
    num_classes = len(le.classes_)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train.values)
    train_data = lgb.Dataset(X_train_final, label=y_train.values,  weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val.values, reference=train_data)

    params = {
        "objective":        "multiclass",
        "num_class":        num_classes,
        "metric":           "multi_logloss",
        "boosting_type":    "gbdt",
        "num_leaves":       63,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples":20,
        "verbose":          -1,
        "seed":             42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # ── Evaluation
    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS")
    print("=" * 60)
    val_probs = model.predict(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    print(classification_report(y_val, val_preds, target_names=le.classes_))

    print("=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    test_probs = model.predict(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    print(classification_report(y_test, test_preds, target_names=le.classes_))

    # ── Save model and label encoder
    import pickle
    model.save_model("C:/Users/Lenovo/Downloads/blood_disorder_lgbm.txt")
    with open("C:/Users/Lenovo/Downloads/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("C:/Users/Lenovo/Downloads/feature_cols.pkl", "wb") as f:
        pickle.dump(list(X_train_final.columns), f)

    print("\nModel saved to: blood_disorder_lgbm.txt")
    print("Label encoder saved to: label_encoder.pkl")
    print("Feature columns saved to: feature_cols.pkl")

    return model, le, X_test, y_test, test_probs

# ─────────────────────────────────────────────
# 8. INFERENCE HELPER
#    Use this at runtime with a real blood report
# ─────────────────────────────────────────────

def predict_single(model, le, feature_cols, input_dict, threshold=0.20):
    """
    input_dict: {feature_name: value} — missing features just omitted
    Returns predicted class and full probability vector.
    threshold: if max prob < threshold, report uncertainty warning.
    """
    # Build row with NaN for missing features
    row = {col: np.nan for col in feature_cols}
    for k, v in input_dict.items():
        if k in row:
            row[k] = v

    # Add presence indicators
    for col in [c for c in feature_cols if not c.endswith("__present")]:
        pres_col = f"{col}__present"
        if pres_col in feature_cols:
            row[pres_col] = 0 if np.isnan(float(row[col]) if row[col] is not None else float('nan')) else 1

    df_input = pd.DataFrame([row])[feature_cols]
    probs = model.predict(df_input)[0]
    pred_idx = np.argmax(probs)
    pred_class = le.classes_[pred_idx]
    confidence = probs[pred_idx]

    result = {
        "predicted_class": pred_class,
        "confidence":       round(float(confidence), 4),
        "all_probabilities": {
            le.classes_[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
        "low_confidence_warning": confidence < threshold,
    }
    return result


if __name__ == "__main__":
    model, le, X_test, y_test, test_probs = main()
