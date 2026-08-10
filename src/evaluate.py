import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.get_data import get_data

def evaluate_models():
    # get test data
    data = get_data()
    X_test = data["X_test"]
    X_test_scaled = data["X_test_scaled"]
    y_test = data["y_test"]

    # download models
    model_v1 = joblib.load("../models/model_v1.pkl")
    model_v2 = joblib.load("../models/model_v2.pkl")

    # evaluate predicts =>  linear regression and calculate MAE & RMSE
    preds_v1 = model_v1.predict(X_test_scaled)
    mae_v1 = mean_absolute_error(y_test, preds_v1)
    rmse_v1 = np.sqrt(mean_squared_error(y_test, preds_v1))

    # evaluate predicts =>  Random Forest and calculate MAE & RMSE
    preds_v2 = model_v2.predict(X_test)
    mae_v2 = mean_absolute_error(y_test, preds_v2)
    rmse_v2 = np.sqrt(mean_squared_error(y_test, preds_v2))


    # *********  Results and Comparisons  **********

    print(f"Model v1 (Linear Regression) -> MAE: {mae_v1:.4f} | RMSE: {rmse_v1:.4f}")
    print(f"Model v2 (Random Forest)    -> MAE: {mae_v2:.4f} | RMSE: {rmse_v2:.4f}")

    if mae_v2 < mae_v1:
        print(f"mae_v2 is {mae_v2:.4f} and it is smaller than from mae_v1 {mae_v1:.4f}")
        print("The error rate is lower in Random Forest Regression")
        print("Random Forest Regression is the best model for house price estimator")
    else:
        print(f"mae_v2 is {mae_v1:.4f} and it is smaller than from mae_v1 {mae_v2:.4f}")
        print("The error rate is lower in Linear Regression")
        print("Linear Regression is the best model for house price estimator")

evaluate_models()

