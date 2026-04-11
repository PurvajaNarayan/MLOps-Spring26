# Lab 6 — Food Inspection Pass/Fail Prediction on GCP Compute Engine

## Objective

This lab demonstrates an end-to-end MLOps workflow: training an **XGBoost classification model** locally to predict whether a Boston food establishment will pass its health inspection, then deploying the model as a **FastAPI REST API** on a **Google Cloud Platform (GCP) Compute Engine** virtual machine.

---

## Project Structure

```
Lab6-VMEngine/
├── README.md                            # This file
├── food_inspection_model.py             # Model training script (run locally)
├── food_inspection_boston_dataset.csv    # Boston food inspection dataset (~880K rows)
├── xgboost_food_inspection.pkl          # Trained XGBoost model artifact
├── app.py                               # FastAPI prediction API
├── confusion_matrix.png                 # Model evaluation — confusion matrix
└── feature_importance.png               # Model evaluation — feature importance plot
```

---

## Part 1: Model Training (Local)

### Dataset

- **Source:** Boston Food Inspection dataset
- **Size:** ~880,000 inspection records
- **Key Columns:** `businessname`, `licstatus`, `licensecat`, `descript`, `result`, `viol_level`, `city`, `zip`, `resultdttm`, `violdttm`

### Target Variable

The `result` column is mapped to binary classification:

| Original Value | Label | Meaning |
|---|---|---|
| `HE_Pass` | **1** (Pass) | Establishment passed inspection |
| `HE_Fail`, `HE_FailExt` | **0** (Fail) | Establishment failed inspection |

After filtering, **721,474 records** were used (61% Fail, 39% Pass).

### Feature Engineering

14 features were engineered from the raw data:

| Feature | Description |
|---|---|
| `is_active` | Whether the license is currently active (binary) |
| `viol_level_num` | Violation severity: `*`→1, `**`→2, `***`→3 |
| `licensecat_enc` | License category (label encoded) |
| `descript_enc` | Business description / type (label encoded) |
| `city_enc` | City name (label encoded) |
| `zip_enc` | Zip code (label encoded) |
| `result_year` | Year of the inspection |
| `result_month` | Month of the inspection |
| `result_dayofweek` | Day of the week (0=Mon, 6=Sun) |
| `result_hour` | Hour of the inspection |
| `license_age_days` | Days between license issue date and inspection |
| `total_inspections` | Total number of inspections per business |
| `business_pass_rate` | Historical pass rate for this specific business |
| `zip_pass_rate` | Average pass rate for businesses in this zip code |

### Model Configuration

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=<auto-calculated for class imbalance>,
    eval_metric="logloss",
)
```

### Model Results

| Metric | Value |
|---|---|
| **Accuracy** | 67.28% |
| **ROC-AUC** | 0.7514 |
| **Fail Precision** | 0.76 |
| **Fail Recall** | 0.68 |
| **Pass Precision** | 0.57 |
| **Pass Recall** | 0.66 |

### How to Run Training Locally

```bash
cd Lab6-VMEngine

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

# Train the model
python food_inspection_model.py
```

This produces:
- `xgboost_food_inspection.pkl` — Serialized trained model
- `confusion_matrix.png` — Confusion matrix heatmap
- `feature_importance.png` — Feature importance bar chart

---

## Part 2: Deployment on GCP Compute Engine

### FastAPI Application (`app.py`)

A lightweight REST API built with FastAPI that serves predictions from the trained model.

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/predict` | Accepts inspection features, returns Pass/Fail prediction with confidence |
| `GET` | `/docs` | Interactive Swagger UI documentation |

**Sample Prediction Response:**
```json
{
    "prediction": "Pass",
    "confidence": 0.732,
    "pass_probability": 0.732
}
```

### GCP VM Setup Steps

#### 1. Create the Compute Engine VM

```bash
gcloud compute instances create food-inspection-api \
    --zone=us-east1-b \
    --machine-type=e2-small \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --tags=http-server
```

**VM Specs:**
| Property | Value |
|---|---|
| Name | `food-inspection-api` |
| Zone | `us-east1-b` |
| Machine Type | `e2-small` |
| OS | Ubuntu 22.04 LTS |
| External IP | `34.23.199.53` |

#### 2. Create Firewall Rule for Port 8000

```bash
gcloud compute firewall-rules create allow-fastapi \
    --allow=tcp:8000 \
    --target-tags=http-server
```

#### 3. SSH into the VM

```bash
gcloud compute ssh food-inspection-api --zone=us-east1-b
```

#### 4. Install Dependencies on the VM

```bash
# System packages
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Create app directory and virtual environment
mkdir ~/app && cd ~/app
python3 -m venv venv
source venv/bin/activate

# Python packages
pip install fastapi uvicorn xgboost scikit-learn pandas joblib
```

#### 5. Transfer Files to the VM (from local machine)

```bash
gcloud compute scp app.py xgboost_food_inspection.pkl \
    purvajanarayana@food-inspection-api:~/app/ --zone=us-east1-b
```

#### 6. Start the API Server on the VM

```bash
cd ~/app
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

#### 7. Access the API

| Resource | URL |
|---|---|
| Swagger Docs | `http://34.23.199.53:8000/docs` |
| Health Check | `http://34.23.199.53:8000/health` |
| Prediction | `POST http://34.23.199.53:8000/predict` |

### Sample cURL Request

```bash
curl -X POST http://34.23.199.53:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "is_active": 1,
    "viol_level_num": 2,
    "licensecat_enc": 5,
    "descript_enc": 3,
    "city_enc": 1,
    "zip_enc": 10,
    "result_year": 2024,
    "result_month": 6,
    "result_dayofweek": 2,
    "result_hour": 14,
    "license_age_days": 1200,
    "total_inspections": 15,
    "business_pass_rate": 0.6,
    "zip_pass_rate": 0.45
  }'
```

---

## Architecture Diagram

```
┌──────────────────┐       ┌──────────────────────────────────────────┐
│   Local Machine  │       │   GCP Compute Engine (e2-small)         │
│                  │       │   Ubuntu 22.04 LTS                      │
│  ┌────────────┐  │  SCP  │  ┌──────────────────────────────────┐   │
│  │ Train      │──┼───────┼─▶│  ~/app/                          │   │
│  │ XGBoost    │  │       │  │  ├── app.py (FastAPI)             │   │
│  │ Model      │  │       │  │  ├── xgboost_food_inspection.pkl │   │
│  └────────────┘  │       │  │  └── venv/                       │   │
│        │         │       │  └──────────────────────────────────┘   │
│        ▼         │       │              │                          │
│  .pkl model file │       │         uvicorn :8000                   │
│                  │       │              │                          │
└──────────────────┘       └──────────────┼──────────────────────────┘
                                          │
                                   Firewall: tcp:8000
                                          │
                                    ┌─────▼─────┐
                                    │  Internet  │
                                    │  Clients   │
                                    └───────────┘
```

---

## Cleanup

To avoid ongoing charges, stop or delete the VM when done:

```bash
# Stop the VM (preserves disk, no compute charges)
gcloud compute instances stop food-inspection-api --zone=us-east1-b

# Or delete the VM entirely
gcloud compute instances delete food-inspection-api --zone=us-east1-b

# Delete the firewall rule
gcloud compute firewall-rules delete allow-fastapi
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10** | Programming language |
| **XGBoost** | Gradient boosting classification model |
| **scikit-learn** | Data preprocessing, train/test split, metrics |
| **pandas / numpy** | Data manipulation and feature engineering |
| **FastAPI** | REST API framework for serving predictions |
| **Uvicorn** | ASGI server for FastAPI |
| **GCP Compute Engine** | Cloud VM hosting the prediction API |
| **matplotlib / seaborn** | Model evaluation visualizations |
| **joblib** | Model serialization (pickle) |
