import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
import pickle

FILES = {
    "Diabetes":               "C:/Users/Lenovo/Downloads/diabetes_csv_updated.csv",
    "Normal":                 "C:/Users/Lenovo/Downloads/All_Normal_csv_updated.csv",
    "Clotting":               "C:/Users/Lenovo/Downloads/Clotting_1000samples_final_csv_updated.csv",
    "Hormonal_Endocrine":     "C:/Users/Lenovo/Downloads/Hormonal_Endocrine_rightmost_only_csv_updated.csv",
    "Infection_Inflammation": "C:/Users/Lenovo/Downloads/Infection_Inflammation_1000samples_final_csv_updated.csv",
    "Lipid_CV":               "C:/Users/Lenovo/Downloads/Lipid_CV_1000samples_final_csv_updated.csv",
    "Liver":                  "C:/Users/Lenovo/Downloads/Liver_1500samples_csv_updated.csv",
    "Nutritional":            "C:/Users/Lenovo/Downloads/Nutritional_1500samples_final_csv_updated.csv",
    "Renal":                  "C:/Users/Lenovo/Downloads/Renal_1000samples_final_csv_updated.csv",
    "Electrolyte":            "C:/Users/Lenovo/Downloads/Electrolyte_1500samples_final_csv_updated.csv",
    "Hematological":          "C:/Users/Lenovo/Downloads/Hematological_2000_formatted_csv_updated.csv",
    "Thyroid":                "C:/Users/Lenovo/Downloads/Thyroid_1000samples_final_csv__1__updated.csv",
}
DROP_COLS = ["sample_id", "disease_focus"]
PANELS = {
    "Diabetes": ["fasting_glucose", "HbA1c", "HOMA_IR", "RBS", "PPG"],
    "Hematological": [
        "hemoglobin", "RBC", "hematocrit", "MCV", "MCH", "MCHC", "RDW",
        "platelets", "neutrophils_pct", "lymphocytes_pct", "monocytes_pct",
        "eosinophils_pct", "basophils_pct"
    ],
    "Thyroid": ["TSH", "Free_T3", "Free_T4", "anti_TPO"],
    "Renal": ["creatinine", "urea_BUN", "eGFR", "uric_acid"],
    "Liver": ["ALT", "AST", "ALP", "GGT", "bilirubin_total", "bilirubin_direct", "albumin", "total_protein"],
    "Lipid_CV": ["total_cholesterol", "LDL", "HDL", "VLDL", "triglycerides", "non_HDL"],
    "Infection_Inflammation": ["WBC", "CRP", "ESR", "procalcitonin"],
    "Nutritional": ["Vitamin_D", "B12", "folate", "iron", "ferritin", "TIBC", "transferrin_sat", "zinc"],
    "Electrolyte": ["sodium", "potassium", "chloride", "bicarbonate", "calcium", "magnesium", "phosphorus", "anion_gap"],
    "Hormonal_Endocrine": ["cortisol", "testosterone", "estrogen", "FSH", "LH", "prolactin", "DHEA_S"],
    "Clotting": ["PT", "INR", "aPTT", "fibrinogen", "D_dimer", "thrombin_time"],
}
ALL_PANELS = list(PANELS.items())
PANEL_DROPOUT_PROB = 0.75
WITHIN_PANEL_DROPOUT = 0.30
def load_and_filter(class_name, filename):
    df = pd.read_csv(filename)
    status_col = df.columns[-1]
    if class_name == "Normal":
        mask = df[status_col].str.contains("normal", case=False, na=False)
    else:
        mask = df[status_col].str.contains("abnormal", case=False, na=False)
    df = df[mask].copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns] + [status_col]
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

def add_presence_indicators(df, feature_cols):
    for col in feature_cols:
        df[f"{col}__present"] = df[col].notna().astype(np.int8)
    return df

def apply_panel_dropout(df, feature_cols, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    df_out = df.copy()
    feature_set = set(feature_cols)
    for idx in df_out.index:
        cls = df_out.at[idx, "label"]
        for panel_name, panel_features in ALL_PANELS:
            present_features = [f for f in panel_features if f in feature_set]
            if not present_features:
                continue
            is_defining = (panel_name == cls)
            if cls == "Normal":
                if rng.random() < PANEL_DROPOUT_PROB:
                    df_out.loc[idx, present_features] = np.nan
                else:
                    for feat in present_features:
                        if rng.random() < WITHIN_PANEL_DROPOUT:
                            df_out.at[idx, feat] = np.nan
            elif is_defining:
                for feat in present_features:
                    if rng.random() < WITHIN_PANEL_DROPOUT:
                        df_out.at[idx, feat] = np.nan
            else:
                if rng.random() < PANEL_DROPOUT_PROB:
                    df_out.loc[idx, present_features] = np.nan
    return df_out

def main():
    df = load_all()
    feature_cols = [c for c in df.columns if c != "label"]
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1}).astype("float32")
    df = add_presence_indicators(df, feature_cols)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    X = df.drop(columns=["label", "label_enc"])
    y = df["label_enc"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    train_labels = df.loc[X_train.index, "label"]
    X_train = X_train.copy()
    X_train["label"] = train_labels.values
    rng = np.random.default_rng(42)
    X_train_dropped = apply_panel_dropout(X_train, feature_cols, rng=rng)
    X_train_final = X_train_dropped.drop(columns=["label"])
    for col in feature_cols:
        pres_col = f"{col}__present"
        if pres_col in X_train_final.columns:
            X_train_final[pres_col] = X_train_final[col].notna().astype(np.int8)
    num_classes = len(le.classes_)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train.values)
    train_data = lgb.Dataset(X_train_final, label=y_train.values, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val.values, reference=train_data)
    params = {
        "objective":         "multiclass",
        "num_class":         num_classes,
        "metric":            "multi_logloss",
        "boosting_type":     "gbdt",
        "num_leaves":        63,
        "learning_rate":     0.05,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "min_child_samples": 20,
        "verbose":           -1,
        "seed":              42,
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50)
        ],
    )
    val_preds = np.argmax(model.predict(X_val), axis=1)
    test_preds = np.argmax(model.predict(X_test), axis=1)
    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nValidation accuracy: {val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    model.save_model("C:/Users/Lenovo/Downloads/blood_disorder_lgbm.txt")
    with open("C:/Users/Lenovo/Downloads/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open("C:/Users/Lenovo/Downloads/feature_cols.pkl", "wb") as f:
        pickle.dump(list(X_train_final.columns), f)

if __name__ == "__main__":
    main()