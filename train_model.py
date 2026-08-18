"""
train_model.py
Feature engineering + GradientBoosting model training for the Predictive Pothole system.
Outputs: model.pkl, feature_names.pkl, feature_importance.csv, label_encoder.pkl
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(OUTPUT_DIR, "pothole_training_data.csv")


# ─── Feature columns used by the model ────────────────────────────────────────
def get_feature_columns():
    return [
        # Weather
        "temperature_2m", "relative_humidity", "dew_point",
        "wind_speed", "max_temp", "min_temp", "avg_temp",
        "precipitation", "pressure",
        # Traffic
        "PCU_City_Wide_Avg_Volume", "Is_Peak_Traffic_Hour", "Is_Weekend",
        # Road static
        "Road_Age_Years", "Drainage_Score", "Subsurface_Quality_Index",
        "Heavy_Vehicle_Ratio", "Last_Repair_Days_Ago", "Surface_Roughness_Index",
        "Nearby_Construction_Flag", "Repair_Backlog_Score", "Lane_Count",
        # Road material one-hot
        "Material_Asphalt", "Material_Concrete", "Material_Interlock",
        # Engineered
        "Cumulative_Rainfall_7d", "Cumulative_Rainfall_30d", "Temp_Swing",
        "Heat_Cycle_Count_30d", "Traffic_Stress_Index", "Water_Logging_Risk",
        "Aging_Factor", "Precipitation_Log", "Humidex",
        # Time
        "Month", "DayOfWeek",
    ]


def load_and_prepare(path: str):
    print(f"  Loading data from {path} ...")
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    # One-hot encode Road_Material
    mat_dummies = pd.get_dummies(df["Road_Material"], prefix="Material")
    for col in ["Material_Asphalt", "Material_Concrete", "Material_Interlock"]:
        if col not in mat_dummies.columns:
            mat_dummies[col] = 0
    df = pd.concat([df, mat_dummies], axis=1)

    # Encode target
    le = LabelEncoder()
    le.fit(["Low", "Medium", "High"])
    df["Target"] = le.transform(df["Pothole_Risk"])

    # Time-based split (last 20% of dates = validation)
    df = df.sort_values("Date")
    split_idx = int(len(df) * 0.80)
    train_df  = df.iloc[:split_idx]
    val_df    = df.iloc[split_idx:]

    features = get_feature_columns()
    # Ensure all feature cols exist
    for f in features:
        if f not in df.columns:
            df[f] = 0
            train_df[f] = 0
            val_df[f]   = 0

    X_train = train_df[features].fillna(0)
    y_train = train_df["Target"]
    X_val   = val_df[features].fillna(0)
    y_val   = val_df["Target"]

    return X_train, y_train, X_val, y_val, features, le


def train(X_train, y_train):
    print("\n  Training GradientBoostingClassifier ...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_val, y_val, le):
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)

    acc   = accuracy_score(y_val, y_pred)
    f1    = f1_score(y_val, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    print(f"\n  ── Validation Metrics ──────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_val, y_pred)
    print(f"  Confusion Matrix (Low / Medium / High):\n{cm}")
    return acc, f1, auc


def save_feature_importance(clf, features: list):
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({"Feature": features, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False)
    path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    fi_df.to_csv(path, index=False)
    print(f"\n  Feature importance saved → {path}")
    print(f"  Top 10 features:")
    for _, r in fi_df.head(10).iterrows():
        bar = "█" * int(r["Importance"] * 100)
        print(f"    {r['Feature']:<35} {r['Importance']:.4f}  {bar}")


def main():
    print("=" * 60)
    print("  Predictive Pothole Formation — Model Training")
    print("=" * 60)

    X_train, y_train, X_val, y_val, features, le = load_and_prepare(DATA_PATH)

    print(f"\n  Train samples : {len(X_train):,}")
    print(f"  Val samples   : {len(X_val):,}")
    print(f"  Features      : {len(features)}")

    clf = train(X_train, y_train)
    acc, f1, auc = evaluate(clf, X_val, y_val, le)
    save_feature_importance(clf, features)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    with open(model_path, "wb") as fp:
        pickle.dump(clf, fp)
    print(f"\n  Model saved → {model_path}")

    # Save feature names
    feat_path = os.path.join(OUTPUT_DIR, "feature_names.pkl")
    with open(feat_path, "wb") as fp:
        pickle.dump(features, fp)
    print(f"  Feature names saved → {feat_path}")

    # Save label encoder
    le_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    with open(le_path, "wb") as fp:
        pickle.dump(le, fp)
    print(f"  Label encoder saved → {le_path}")

    print(f"\n{'='*60}")
    print("  ✓ Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()



    