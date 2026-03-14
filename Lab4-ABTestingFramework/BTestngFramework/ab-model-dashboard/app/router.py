"""
ABRouter — Traffic splitting logic for A/B model testing.

Routes incoming prediction requests between Champion and Challenger
models based on configurable traffic weights.
"""

import random
import time
import yaml
import os


class ABRouter:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "config.yaml"
            )
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """Load traffic weights from config file."""
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        self.champion_weight = self.config["traffic"]["champion_pct"] / 100

    def reload_config(self):
        """Reload configuration (for live weight updates)."""
        self._load_config()

    def update_weights(self, champion_pct, challenger_pct):
        """Update traffic weights at runtime."""
        self.config["traffic"]["champion_pct"] = champion_pct
        self.config["traffic"]["challenger_pct"] = challenger_pct
        self.champion_weight = champion_pct / 100

    def route(self):
        """
        Decide which model handles the request.

        Returns:
            str: 'champion' or 'challenger'
        """
        return "champion" if random.random() < self.champion_weight else "challenger"

    def predict(self, model_a, model_b, input_data):
        """
        Route input to a model, measure latency, return result.

        Args:
            model_a: Champion model (must have .predict())
            model_b: Challenger model (must have .predict())
            input_data: DataFrame or array-like input

        Returns:
            dict with prediction, model name, latency_ms, timestamp
        """
        target = self.route()
        model = model_a if target == "champion" else model_b

        start = time.perf_counter()
        prediction = model.predict(input_data)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "prediction": prediction.tolist(),
            "model": target,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
        }

    def get_weights(self):
        """Return current traffic weights."""
        return {
            "champion_pct": self.config["traffic"]["champion_pct"],
            "challenger_pct": self.config["traffic"]["challenger_pct"],
        }
