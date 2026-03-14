"""
Train and register two models for A/B testing.

Trains an ElasticNet (Champion) and RandomForest (Challenger) on the
wine quality dataset, logs them to MLflow, and registers them in the
model registry with appropriate stages.
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.tracking import MlflowClient


def load_wine_data():
    """Load wine quality dataset from UCI repository or local file."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "winequality-red.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, sep=";")
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=";")
        # Save locally for future use
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, sep=";", index=False)
    return df


def eval_metrics(actual, predicted):
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2


def train_champion(X_train, X_test, y_train, y_test):
    """Train ElasticNet as the Champion model."""
    alpha = 0.5
    l1_ratio = 0.5

    with mlflow.start_run(run_name="Champion-ElasticNet") as run:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)

        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio, "model_type": "ElasticNet"})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="WineQualityModel"
        )

        print(f"Champion (ElasticNet) — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return run.info.run_id


def train_challenger(X_train, X_test, y_train, y_test):
    """Train RandomForest as the Challenger model."""
    n_estimators = 100
    max_depth = 10

    with mlflow.start_run(run_name="Challenger-RandomForest") as run:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "model_type": "RandomForest"
        })
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="WineQualityChallenger"
        )

        print(f"Challenger (RandomForest) — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        return run.info.run_id


def promote_models():
    """Transition Champion to Production and Challenger to Staging."""
    client = MlflowClient()

    # Get the latest version of each model
    champion_versions = client.get_latest_versions("WineQualityModel")
    challenger_versions = client.get_latest_versions("WineQualityChallenger")

    if champion_versions:
        latest_champion = champion_versions[0].version
        client.transition_model_version_stage(
            name="WineQualityModel",
            version=latest_champion,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Champion v{latest_champion} → Production")

    if challenger_versions:
        latest_challenger = challenger_versions[0].version
        client.transition_model_version_stage(
            name="WineQualityChallenger",
            version=latest_challenger,
            stage="Staging",
            archive_existing_versions=True
        )
        print(f"Challenger v{latest_challenger} → Staging")


def main():
    # Set MLflow tracking URI (use env var or default to local)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("wine-quality-ab-test")

    print("Loading wine quality data...")
    df = load_wine_data()

    # Separate features and target
    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print()

    print("Training Champion (ElasticNet)...")
    train_champion(X_train, X_test, y_train, y_test)
    print()

    print("Training Challenger (RandomForest)...")
    train_challenger(X_train, X_test, y_train, y_test)
    print()

    print("Promoting models in registry...")
    promote_models()
    print()

    print("✅ Both models trained and registered successfully!")


if __name__ == "__main__":
    main()
