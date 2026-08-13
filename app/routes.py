import joblib
import pandas as pd
from app.schemas import HouseFeatures, PredictionResponse
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

router = APIRouter()

# load models
models = {
    "v1": joblib.load(BASE_DIR / "models" / "model_v1.pkl"),
    "v2": joblib.load(BASE_DIR / "models" / "model_v2.pkl"),
}
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "California Housing Predictor"}

@router.get("/versions")
def get_versions():
    return {"available_versions": ["v1", "v2"], "promoted_live": "v2"}

@router.get("/metadata")
def get_metadata():
    return {
        "model_v1": "Linear Regression (Linear Baseline)",
        "model_v2": "Random Forest Regressor (Promoted Live)",
        "target": "Median House Value in Dollars",
    }

@router.post("/predict", response_model=PredictionResponse)
def predict(
    data: HouseFeatures,
    version: str = Query(
        "v2", description="v1 or v2"
    ),
):
    if version not in models:
        raise HTTPException(
            status_code=400,
            detail="version not found",
        )

    # To DataFrame
    input_df = pd.DataFrame([data.dict()])

    # if model_v1 => scaling
    if version == "v1":
        input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    prediction = models[version].predict(input_df)[0]

    # value * 100000
    actual_price = round(float(prediction) * 100000, 2)

    return {
        "predicted_price_dollars": actual_price,
        "model_version": version,
    }
