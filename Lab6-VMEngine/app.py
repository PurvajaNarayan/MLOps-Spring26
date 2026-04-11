from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Food Inspection Predictor")

model = joblib.load("xgboost_food_inspection.pkl")

class InspectionInput(BaseModel):
    is_active: int
    viol_level_num: int
    licensecat_enc: int
    descript_enc: int
    city_enc: int
    zip_enc: int
    result_year: int
    result_month: int
    result_dayofweek: int
    result_hour: int
    license_age_days: float
    total_inspections: int
    business_pass_rate: float
    zip_pass_rate: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InspectionInput):
    features = pd.DataFrame([data.model_dump()])
    prob = model.predict_proba(features)[0]
    prediction = int(model.predict(features)[0])
    return {
        "prediction": "Pass" if prediction == 1 else "Fail",
        "confidence": round(float(max(prob)), 3),
        "pass_probability": round(float(prob[1]), 3),
    }