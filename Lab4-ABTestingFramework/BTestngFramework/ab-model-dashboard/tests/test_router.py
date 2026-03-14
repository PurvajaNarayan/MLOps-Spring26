"""Tests for the ABRouter traffic splitting logic."""

import pytest
import os
import tempfile
import yaml
from unittest.mock import MagicMock
import numpy as np


# Adjust path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.router import ABRouter


@pytest.fixture
def config_file():
    """Create a temporary config file for testing."""
    config = {
        "traffic": {"champion_pct": 70, "challenger_pct": 30},
        "models": {
            "champion": {"name": "TestChampion", "stage": "Production"},
            "challenger": {"name": "TestChallenger", "stage": "Staging"},
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def router(config_file):
    """Create an ABRouter with the test config."""
    return ABRouter(config_path=config_file)


class TestABRouter:
    def test_init_loads_config(self, router):
        """Router should load champion weight from config."""
        assert router.champion_weight == 0.7

    def test_route_returns_valid_model(self, router):
        """route() should always return 'champion' or 'challenger'."""
        for _ in range(100):
            result = router.route()
            assert result in ("champion", "challenger")

    def test_traffic_distribution_approximate(self, router):
        """With 70/30 split, roughly 70% should go to champion over many trials."""
        n = 10000
        results = [router.route() for _ in range(n)]
        champion_ratio = results.count("champion") / n
        # Allow 5% tolerance
        assert 0.65 <= champion_ratio <= 0.75, f"Got {champion_ratio:.2%} champion"

    def test_100_percent_champion(self, config_file):
        """100/0 split should always route to champion."""
        router = ABRouter(config_path=config_file)
        router.update_weights(100, 0)
        results = [router.route() for _ in range(100)]
        assert all(r == "champion" for r in results)

    def test_0_percent_champion(self, config_file):
        """0/100 split should always route to challenger."""
        router = ABRouter(config_path=config_file)
        router.update_weights(0, 100)
        results = [router.route() for _ in range(100)]
        assert all(r == "challenger" for r in results)

    def test_update_weights(self, router):
        """update_weights should change the routing distribution."""
        router.update_weights(50, 50)
        assert router.champion_weight == 0.5
        assert router.config["traffic"]["champion_pct"] == 50

    def test_get_weights(self, router):
        """get_weights should return current percentages."""
        weights = router.get_weights()
        assert weights["champion_pct"] == 70
        assert weights["challenger_pct"] == 30

    def test_predict_returns_correct_structure(self, router):
        """predict() should return dict with prediction, model, latency_ms, timestamp."""
        mock_model_a = MagicMock()
        mock_model_b = MagicMock()
        mock_model_a.predict.return_value = np.array([5.5])
        mock_model_b.predict.return_value = np.array([6.2])

        input_data = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.998, 3.51, 0.56, 9.4]]
        result = router.predict(mock_model_a, mock_model_b, input_data)

        assert "prediction" in result
        assert "model" in result
        assert "latency_ms" in result
        assert "timestamp" in result
        assert result["model"] in ("champion", "challenger")
        assert isinstance(result["latency_ms"], float)
        assert isinstance(result["prediction"], list)

    def test_predict_calls_correct_model(self, config_file):
        """predict() should call the model corresponding to the routing decision."""
        router = ABRouter(config_path=config_file)
        router.update_weights(100, 0)  # Always champion

        mock_a = MagicMock()
        mock_b = MagicMock()
        mock_a.predict.return_value = np.array([5.0])
        mock_b.predict.return_value = np.array([6.0])

        result = router.predict(mock_a, mock_b, [[1, 2, 3]])
        assert result["model"] == "champion"
        mock_a.predict.assert_called_once()
        mock_b.predict.assert_not_called()

    def test_predict_measures_latency(self, router):
        """Latency should be a positive number."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.0])

        result = router.predict(mock_model, mock_model, [[1, 2, 3]])
        assert result["latency_ms"] >= 0
