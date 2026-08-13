import joblib
import pandas as pd
from app.schemas import HouseFeatures, PredictionResponse
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

# load models
models = {
    "v1": joblib.load("models/model_v1.pkl"),
    "v2": joblib.load("models/model_v2.pkl"),
}
scaler = joblib.load("models/scaler.pkl")

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
