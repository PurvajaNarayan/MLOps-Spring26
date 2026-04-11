"""
Food Inspection Pass/Fail Prediction using XGBoost
===================================================
Predicts whether a food establishment in Boston will pass its health inspection.

Target: 'result' column → binary: Pass (1) vs Fail (0)
Features engineered from license status, category, description, violation level,
city, zip code, and temporal features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "food_inspection_boston_dataset.csv")
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Raw dataset shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. Define the target variable
# ─────────────────────────────────────────────
# Map 'result' to binary: Pass = 1, Fail = 0
# Keep only clear Pass/Fail results for clean modelling
pass_labels = ["HE_Pass"]
fail_labels = ["HE_Fail", "HE_FailExt"]

df = df[df["result"].isin(pass_labels + fail_labels)].copy()
df["target"] = df["result"].apply(lambda x: 1 if x in pass_labels else 0)

print(f"\n  Filtered dataset shape (Pass/Fail only): {df.shape}")
print(f"  Target distribution:\n{df['target'].value_counts()}")
print(f"  Pass rate: {df['target'].mean():.2%}")

# ─────────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────────
print("\nEngineering features...")

# Parse datetime columns
for col in ["issdttm", "expdttm", "resultdttm", "violdttm"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# Temporal features from the result date
df["result_year"] = df["resultdttm"].dt.year
df["result_month"] = df["resultdttm"].dt.month
df["result_dayofweek"] = df["resultdttm"].dt.dayofweek
df["result_hour"] = df["resultdttm"].dt.hour

# License age in days (from issue date to result date)
df["license_age_days"] = (df["resultdttm"] - df["issdttm"]).dt.days

# Whether the license is active or not
df["is_active"] = (df["licstatus"] == "Active").astype(int)

# Violation level encoding (* = 1, ** = 2, *** = 3, else 0)
viol_map = {"*": 1, "**": 2, "***": 3}
df["viol_level_num"] = df["viol_level"].map(viol_map).fillna(0).astype(int)

# Encode categorical columns
label_encoders = {}
cat_cols = ["licensecat", "descript", "city", "zip"]

for col in cat_cols:
    df[col] = df[col].astype(str).fillna("Unknown")
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    label_encoders[col] = le

# Historical violation count per business (as a proxy for repeat offenders)
viol_counts = df.groupby("licenseno").size().reset_index(name="total_inspections")
df = df.merge(viol_counts, on="licenseno", how="left")

# Historical fail rate per business
fail_rate = (
    df.groupby("licenseno")["target"]
    .mean()
    .reset_index(name="business_pass_rate")
)
df = df.merge(fail_rate, on="licenseno", how="left")

# Zip-code level pass rate
zip_rate = (
    df.groupby("zip")["target"]
    .mean()
    .reset_index(name="zip_pass_rate")
)
df = df.merge(zip_rate, on="zip", how="left")

# ─────────────────────────────────────────────
# 4. Select features and prepare for modelling
# ─────────────────────────────────────────────
feature_cols = [
    "is_active",
    "viol_level_num",
    "licensecat_enc",
    "descript_enc",
    "city_enc",
    "zip_enc",
    "result_year",
    "result_month",
    "result_dayofweek",
    "result_hour",
    "license_age_days",
    "total_inspections",
    "business_pass_rate",
    "zip_pass_rate",
]

X = df[feature_cols].copy()
y = df["target"].copy()

# Handle any remaining NaN values
X = X.fillna(0)

print(f"\n  Feature matrix shape: {X.shape}")
print(f"  Features: {feature_cols}")

# ─────────────────────────────────────────────
# 5. Train / Test split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train set: {X_train.shape[0]} samples")
print(f"  Test  set: {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# 6. Train XGBoost model
# ─────────────────────────────────────────────
print("\nTraining XGBoost model...")

# Calculate scale_pos_weight for class imbalance
n_fail = (y_train == 0).sum()
n_pass = (y_train == 1).sum()
scale_pos_weight = n_fail / n_pass

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# ─────────────────────────────────────────────
# 7. Evaluate the model
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("  MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"\n  Accuracy : {accuracy:.4f}")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# ─────────────────────────────────────────────
# 8. Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Food Inspection Pass/Fail")
plt.tight_layout()
cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\n  Confusion matrix saved → {cm_path}")

# ─────────────────────────────────────────────
# 9. Feature Importance
# ─────────────────────────────────────────────
importances = model.feature_importances_
feat_imp = pd.DataFrame(
    {"Feature": feature_cols, "Importance": importances}
).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(feat_imp["Feature"], feat_imp["Importance"], color="steelblue")
plt.xlabel("Feature Importance (Gain)")
plt.title("XGBoost Feature Importance — Food Inspection Prediction")
plt.tight_layout()
fi_path = os.path.join(os.path.dirname(__file__), "feature_importance.png")
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"  Feature importance plot saved → {fi_path}")

# ─────────────────────────────────────────────
# 10. Save the trained model
# ─────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "xgboost_food_inspection.pkl")
joblib.dump(model, model_path)
print(f"  Trained model saved → {model_path}")

print("\n✅ Done! Model training and evaluation complete.")
