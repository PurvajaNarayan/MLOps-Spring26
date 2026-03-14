# A/B Model Testing Dashboard

A Flask API that routes prediction requests between two MLflow-registered models (Champion vs Challenger), logs metrics, and serves a live Chart.js dashboard for real-time model comparison.

## Architecture

```
                         ┌─────────────────────┐
   Client Request ──────►│   Flask API Gateway  │
                         │  (Traffic Router)    │
                         └────┬───────────┬─────┘
                              │           │
                         70% traffic  30% traffic
                              │           │
                    ┌─────────▼──┐  ┌─────▼────────┐
                    │  Model A   │  │   Model B     │
                    │ (Champion) │  │ (Challenger)  │
                    └─────┬──────┘  └──────┬────────┘
                          │                │
                          ▼                ▼
                   ┌──────────────────────────┐
                   │   MLflow Tracking Server  │
                   │  (logs predictions,       │
                   │   latency, outcomes)      │
                   └──────────┬───────────────┘
                              │
                   ┌──────────▼───────────────┐
                   │   Dashboard (Chart.js)    │
                   │  - Latency comparison     │
                   │  - Prediction distribution│
                   │  - Accuracy over time     │
                   │  - Traffic split view     │
                   └──────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 5001
```

### 3. Train & Register Models

```bash
python models/train_models.py
```

This trains an ElasticNet (Champion) and RandomForest (Challenger) on the wine quality dataset, registers both in MLflow, and promotes Champion → Production, Challenger → Staging.

### 4. Start the Flask App

```bash
python -m flask --app app run --port 8000
```

### 5. Open the Dashboard

Navigate to [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

### 6. Simulate Traffic

```bash
python scripts/simulate_traffic.py --count 500 --rps 10
```

Watch the dashboard update in real time!

---

## Docker Setup

```bash
docker-compose up --build
```

This starts both the MLflow server (port 5001) and the Flask app (port 8000).

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Send features, get prediction + model metadata |
| `/feedback` | POST | Send ground truth for a previous prediction |
| `/dashboard` | GET | Live comparison dashboard |
| `/api/metrics` | GET | JSON metrics (polled by dashboard) |
| `/config` | POST | Update traffic weights at runtime |
| `/health` | GET | Health check |

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
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
      "alcohol": 9.4
    }
  }'
```

**Response:**
```json
{
  "request_id": "a1b2c3d4-...",
  "prediction": [5.56],
  "model": "champion",
  "latency_ms": 2.34,
  "timestamp": 1710000000.0
}
```

### POST /feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "a1b2c3d4-...", "actual": 6.0}'
```

### POST /config

```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"champion_pct": 50, "challenger_pct": 50}'
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
ab-model-dashboard/
├── app/
│   ├── __init__.py           # Flask app factory
│   ├── routes.py             # API endpoints
│   ├── router.py             # Traffic splitting logic
│   ├── model_loader.py       # Load models from MLflow / fallback
│   ├── tracker.py            # Log predictions to MLflow
│   ├── metrics_store.py      # In-memory metrics buffer
│   └── templates/
│       └── dashboard.html    # Live Chart.js dashboard
├── models/
│   └── train_models.py       # Train & register two models
├── scripts/
│   ├── simulate_traffic.py   # Traffic simulator
│   └── send_feedback.py      # Ground truth sender
├── tests/
│   ├── test_router.py
│   ├── test_predictions.py
│   └── test_metrics.py
├── config.yaml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Configuration

Edit `config.yaml` to adjust traffic weights, model names, and monitoring settings:

```yaml
traffic:
  champion_pct: 70
  challenger_pct: 30

models:
  champion:
    name: "WineQualityModel"
    stage: "Production"
  challenger:
    name: "WineQualityChallenger"
    stage: "Staging"
```

Traffic weights can also be changed at runtime via the `/config` endpoint or the dashboard slider.

---

## How This Differs from [MLflow Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1)

The course's MLflow Lab 1 teaches the **foundations** — experiment tracking, model logging, and the model registry. This project takes those building blocks and constructs a **production-grade A/B testing system** on top of them. Here's how they compare:

| Aspect | MLflow Lab 1 (Course) | Lab 4 — A/B Testing Dashboard (This Project) |
|---|---|---|
| **Scope** | Single-model training & logging | Two-model comparison system with live routing |
| **MLflow Usage** | `log_param`, `log_metric`, `log_model` basics | Model Registry stages (Production/Staging), step-level time-series metrics, background thread logging |
| **Models** | One ElasticNet on wine quality | ElasticNet (Champion) + RandomForest (Challenger) on the same dataset |
| **Serving** | `mlflow models serve` (MLflow's built-in server) | Custom Flask API with traffic routing, latency measurement, and request-level metadata |
| **Evaluation** | Offline only (RMSE/MAE/R2 on a test set) | **Online evaluation** — real-time latency tracking, prediction distributions, and accuracy via delayed ground truth feedback |
| **Model Registry** | Registers a model; no stage management | Registers two models, transitions Champion → Production and Challenger → Staging, designed for auto-promotion |
| **Observability** | MLflow UI only | Live Chart.js dashboard with 4 panels (latency, traffic split, distributions, predictions over time) + MLflow logs |
| **Traffic Management** | N/A (single model) | Configurable weighted A/B split (70/30 default), runtime adjustable via API or dashboard slider |
| **Architecture** | Single Python script | Modular Flask app (router, tracker, metrics store, model loader) + Docker Compose + unit tests |
| **Testing** | None | 33 unit tests covering routing logic, API endpoints, and metrics computation |
