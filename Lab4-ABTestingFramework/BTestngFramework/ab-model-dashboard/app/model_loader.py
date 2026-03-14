"""
Model loader — Loads models from MLflow registry or local fallback.
"""

import os
import mlflow.pyfunc
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import pickle


def load_models(config):
    """
    Load Champion and Challenger models from MLflow registry.

    Falls back to locally-saved sklearn models if MLflow is unreachable
    (useful for development and testing without a running MLflow server).

    Args:
        config: dict with models.champion and models.challenger settings

    Returns:
        tuple: (champion_model, challenger_model)
    """
    champion_config = config["models"]["champion"]
    challenger_config = config["models"]["challenger"]

    champion = None
    challenger = None

    # Try loading from MLflow registry
    try:
        champion_uri = f"models:/{champion_config['name']}/{champion_config['stage']}"
        champion = mlflow.pyfunc.load_model(champion_uri)
        print(f"✅ Champion loaded from MLflow: {champion_uri}")
    except Exception as e:
        print(f"⚠️  Could not load Champion from MLflow: {e}")

    try:
        challenger_uri = f"models:/{challenger_config['name']}/{challenger_config['stage']}"
        challenger = mlflow.pyfunc.load_model(challenger_uri)
        print(f"✅ Challenger loaded from MLflow: {challenger_uri}")
    except Exception as e:
        print(f"⚠️  Could not load Challenger from MLflow: {e}")

    # Fallback: load from local pickle files if available
    fallback_dir = os.path.join(os.path.dirname(__file__), "..", "models", "fallback")
    if champion is None:
        champion = _load_fallback(fallback_dir, "champion.pkl")
    if challenger is None:
        challenger = _load_fallback(fallback_dir, "challenger.pkl")

    # Last resort: create simple dummy models for development
    if champion is None:
        print("⚠️  Using dummy ElasticNet as Champion (dev mode)")
        champion = _create_dummy_elasticnet()
    if challenger is None:
        print("⚠️  Using dummy RandomForest as Challenger (dev mode)")
        challenger = _create_dummy_random_forest()

    return champion, challenger


def _load_fallback(fallback_dir, filename):
    """Load a model from a local pickle file."""
    path = os.path.join(fallback_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Loaded fallback model from {path}")
        return model
    return None


def _create_dummy_elasticnet():
    """Create a minimally-trained ElasticNet for development."""
    import numpy as np
    model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    # Train on tiny synthetic data so .predict() works
    X = np.random.RandomState(42).rand(20, 11)
    y = np.random.RandomState(42).rand(20) * 3 + 4
    model.fit(X, y)
    return model


def _create_dummy_random_forest():
    """Create a minimally-trained RandomForest for development."""
    import numpy as np
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    X = np.random.RandomState(42).rand(20, 11)
    y = np.random.RandomState(42).rand(20) * 3 + 4
    model.fit(X, y)
    return model
