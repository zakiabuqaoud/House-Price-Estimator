from app.routes import router
from fastapi import FastAPI

app = FastAPI(
    title="California Housing Prediction API",
    version="1.0.0",
)

app.include_router(router)