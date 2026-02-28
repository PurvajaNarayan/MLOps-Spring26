import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── 1. Load raw transactions ──────────────────────────────────────────────────
df = pd.read_json("transactions.jsonl", lines=True)

# ── 2. Simulate session features by grouping on user_id ───────────────────────
# (Mimics what the Beam session window computes at pipeline runtime)
session_features = df.groupby("user_id").agg(
    session_txn_count=("amount", "count"),
    total_amount=("amount", "sum"),
    avg_amount=("amount", "mean"),
    max_amount=("amount", "max"),
    amount_std=("amount", "std"),
    unique_categories=("merchant_category", "nunique"),
    unique_locations=("location", "nunique"),
).reset_index()

session_features["amount_std"] = session_features["amount_std"].fillna(0)

# Compute location_switches per user
def count_location_switches(locs):
    locs = list(locs)
    return sum(1 for i in range(1, len(locs)) if locs[i] != locs[i - 1])

switches = df.groupby("user_id")["location"].apply(count_location_switches).reset_index()
switches.columns = ["user_id", "location_switches"]

# Compute high_value_txn_ratio per user
def high_value_ratio(amounts):
    amounts = list(amounts)
    return sum(1 for a in amounts if a > 1000) / len(amounts)

hvr = df.groupby("user_id")["amount"].apply(high_value_ratio).reset_index()
hvr.columns = ["user_id", "high_value_txn_ratio"]

# Merge session features back to user level
session_features = session_features.merge(switches, on="user_id").merge(hvr, on="user_id")

# ── 3. Merge back with transaction-level data for labels ─────────────────────
# Use per-transaction features + session features joined on user_id
df = df.merge(session_features, on="user_id", suffixes=("", "_session"))

# ── 4. Define features and label ──────────────────────────────────────────────
feature_cols = [
    "amount", "session_txn_count", "total_amount",
    "avg_amount", "max_amount", "amount_std",
    "unique_categories", "unique_locations",
    "location_switches", "high_value_txn_ratio"
]

X = df[feature_cols].fillna(0)
y = df["is_fraud"]

print(f"Dataset: {len(df)} transactions | Fraud rate: {y.mean():.2%}")

# ── 5. Train ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

# ── 6. Save model ─────────────────────────────────────────────────────────────
import os
os.makedirs("model", exist_ok=True)
with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to model/fraud_model.pkl")