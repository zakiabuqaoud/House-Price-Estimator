
from pydantic import BaseModel, Field

class HouseFeatures(BaseModel):
    MedInc: float = Field(..., example=8.32, description="Income Mediam")
    HouseAge: float = Field(..., example=41.0, description="house age")
    AveRooms: float = Field(..., example=6.98, description="number of rooms")
    AveBedrms: float = Field(..., example=1.02, description="number of bedroooms")
    Population: float = Field(..., example=322.0, description="Population")
    AveOccup: float = Field(..., example=2.55, description="number of residence")
    Latitude: float = Field(..., example=37.88, description="Latitude")
    Longitude: float = Field(..., example=-122.23, description="Longitude")

class PredictionResponse(BaseModel):
    predicted_price_dollars: float
    model_version: str