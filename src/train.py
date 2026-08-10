
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.get_data import get_data

def train_models():
    data = get_data()
    # Model_v1 is Linear Regression is Started
    print("Linear Regression v1 is training ...")
    model_v1 = LinearRegression()
    model_v1.fit(data["X_train_scaled"], data["y_train"])
    joblib.dump(model_v1, "../models/model_v1.pkl")
    joblib.dump(data["scaler"], "../models/scaler.pkl")
    print("model_v1 is saved in path (../models/model_v1.pkl)")
    print("model_v1 Training is Finished")

    # Model_v2 is Random Forest Regressor is Started
    print("Random Forest Regressor v2 is training ...")
    model_v2 = RandomForestRegressor(n_estimators=100, random_state=42)
    model_v2.fit(data["X_train"], data["y_train"])
    joblib.dump(model_v2, "../models/model_v2.pkl")
    print("model_v2 is saved in path (../models/model_v2.pkl)")
    print("model_v2 Training is Finished")

train_models()
