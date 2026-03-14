"""
Flask API routes for the A/B Model Testing Dashboard.

Endpoints:
  POST /predict     — Route input to a model, return prediction + metadata
  POST /feedback    — Accept ground truth for a prior prediction
  GET  /dashboard   — Serve the live Chart.js dashboard
  GET  /api/metrics — JSON metrics endpoint polled by dashboard
  POST /config      — Update traffic weights at runtime
"""

import uuid
import pandas as pd
from flask import Blueprint, request, jsonify, render_template, current_app

bp = Blueprint("main", __name__)


@bp.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Expects JSON:
      { "features": { "fixed acidity": 7.4, "volatile acidity": 0.7, ... } }

    Returns JSON:
      { "request_id": "...", "prediction": [...], "model": "champion"|"challenger",
        "latency_ms": 2.34, "timestamp": 1234567890.0 }
    """
    data = request.json
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    input_df = pd.DataFrame([data["features"]])

    result = current_app.router.predict(
        current_app.champion, current_app.challenger, input_df
    )
    result["request_id"] = str(uuid.uuid4())

    # Log to MLflow (fire-and-forget)
    current_app.tracker.log_prediction(result)

    # Buffer for dashboard
    current_app.metrics_store.record(result)

    return jsonify(result)


@bp.route("/feedback", methods=["POST"])
def feedback():
    """
    Accept ground truth for a previous prediction.

    Expects JSON:
      { "request_id": "...", "actual": 6.0 }
    """
    data = request.json
    if not data or "request_id" not in data or "actual" not in data:
        return jsonify({"error": "Missing 'request_id' or 'actual'"}), 400

    request_id = data["request_id"]
    actual_value = float(data["actual"])

    current_app.metrics_store.record_outcome(request_id, actual_value)
    current_app.tracker.log_outcome(request_id, actual_value)

    return jsonify({"status": "recorded", "request_id": request_id})


@bp.route("/dashboard")
def dashboard():
    """Serve the live A/B testing dashboard."""
    refresh_ms = (
        current_app.config.get("AB_CONFIG", {})
        .get("monitoring", {})
        .get("dashboard_refresh_ms", 2000)
    )
    return render_template("dashboard.html", refresh_ms=refresh_ms)


@bp.route("/api/metrics")
def api_metrics():
    """JSON endpoint the dashboard polls for live data."""
    summary = current_app.metrics_store.get_summary()
    weights = current_app.router.get_weights()
    summary["weights"] = weights
    return jsonify(summary)


@bp.route("/config", methods=["POST"])
def update_config():
    """
    Update traffic weights at runtime.

    Expects JSON:
      { "champion_pct": 50, "challenger_pct": 50 }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    champion_pct = data.get("champion_pct")
    challenger_pct = data.get("challenger_pct")

    if champion_pct is None or challenger_pct is None:
        return jsonify({"error": "Provide both 'champion_pct' and 'challenger_pct'"}), 400

    if champion_pct + challenger_pct != 100:
        return jsonify({"error": "Percentages must sum to 100"}), 400

    current_app.router.update_weights(champion_pct, challenger_pct)

    return jsonify({
        "status": "updated",
        "champion_pct": champion_pct,
        "challenger_pct": challenger_pct,
    })


@bp.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "ab-model-dashboard"})
