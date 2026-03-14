"""Tests for the prediction endpoint."""

import pytest
import json
import os
import sys
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import create_app


@pytest.fixture
def config_file():
    """Create a temporary config for testing."""
    config = {
        "traffic": {"champion_pct": 70, "challenger_pct": 30},
        "models": {
            "champion": {"name": "TestChampion", "stage": "Production"},
            "challenger": {"name": "TestChallenger", "stage": "Staging"},
        },
        "monitoring": {"buffer_size": 100, "dashboard_refresh_ms": 2000},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def client(config_file):
    """Create a Flask test client with dummy models."""
    # Set env to avoid MLflow connection
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:99999"
    app = create_app(config_path=config_file)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


SAMPLE_FEATURES = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        """POST /predict with valid features should return 200."""
        resp = client.post(
            "/predict",
            data=json.dumps({"features": SAMPLE_FEATURES}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_predict_response_structure(self, client):
        """Response should have prediction, model, latency_ms, request_id, timestamp."""
        resp = client.post(
            "/predict",
            data=json.dumps({"features": SAMPLE_FEATURES}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "prediction" in data
        assert "model" in data
        assert "latency_ms" in data
        assert "request_id" in data
        assert "timestamp" in data

    def test_predict_model_is_valid(self, client):
        """model field should be 'champion' or 'challenger'."""
        resp = client.post(
            "/predict",
            data=json.dumps({"features": SAMPLE_FEATURES}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert data["model"] in ("champion", "challenger")

    def test_predict_missing_features(self, client):
        """POST /predict without 'features' should return 400."""
        resp = client.post(
            "/predict",
            data=json.dumps({"wrong_key": {}}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_predict_empty_body(self, client):
        """POST /predict with empty body should return 400."""
        resp = client.post(
            "/predict",
            data="",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_predict_request_id_unique(self, client):
        """Each prediction should get a unique request_id."""
        ids = set()
        for _ in range(10):
            resp = client.post(
                "/predict",
                data=json.dumps({"features": SAMPLE_FEATURES}),
                content_type="application/json",
            )
            data = resp.get_json()
            ids.add(data["request_id"])
        assert len(ids) == 10


class TestFeedbackEndpoint:
    def test_feedback_returns_200(self, client):
        """POST /feedback with valid data should return 200."""
        # First make a prediction
        resp = client.post(
            "/predict",
            data=json.dumps({"features": SAMPLE_FEATURES}),
            content_type="application/json",
        )
        request_id = resp.get_json()["request_id"]

        # Send feedback
        resp = client.post(
            "/feedback",
            data=json.dumps({"request_id": request_id, "actual": 6.0}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "recorded"

    def test_feedback_missing_fields(self, client):
        """POST /feedback without required fields should return 400."""
        resp = client.post(
            "/feedback",
            data=json.dumps({"request_id": "abc"}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestConfigEndpoint:
    def test_config_update_returns_200(self, client):
        """POST /config should update weights."""
        resp = client.post(
            "/config",
            data=json.dumps({"champion_pct": 50, "challenger_pct": 50}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["champion_pct"] == 50
        assert data["challenger_pct"] == 50

    def test_config_invalid_sum(self, client):
        """Percentages not summing to 100 should return 400."""
        resp = client.post(
            "/config",
            data=json.dumps({"champion_pct": 60, "challenger_pct": 60}),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestDashboardAndHealth:
    def test_dashboard_returns_html(self, client):
        """GET /dashboard should return HTML."""
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert b"Chart.js" in resp.data or b"chart" in resp.data.lower()

    def test_api_metrics_returns_json(self, client):
        """GET /api/metrics should return valid JSON summary."""
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "champion" in data
        assert "challenger" in data
        assert "traffic_split" in data

    def test_health_check(self, client):
        """GET /health should return healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "healthy"
