"""
Flask application factory for the A/B Model Testing Dashboard.
"""

import os
import yaml
from flask import Flask

from .router import ABRouter
from .model_loader import load_models
from .tracker import ABTracker
from .metrics_store import MetricsStore


def create_app(config_path=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Resolve config path
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    app.config["AB_CONFIG"] = config
    app.config["CONFIG_PATH"] = config_path

    # Initialize components
    app.champion, app.challenger = load_models(config)
    app.router = ABRouter(config_path)
    app.tracker = ABTracker()
    app.metrics_store = MetricsStore(
        max_size=config.get("monitoring", {}).get("buffer_size", 1000)
    )

    # Register routes
    from .routes import bp
    app.register_blueprint(bp)

    print("🚀 A/B Model Testing Dashboard is ready!")
    return app
