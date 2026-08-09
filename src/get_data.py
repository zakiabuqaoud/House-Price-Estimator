
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 20% Testing , 80% training
# random_state=42 => same result every time
def get_data(test_size=0.2, random_state=42):
    # 1- get Data from sklearn Library
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target


    # 2- Split Data to train and test group
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # Feature Scaling For prevent Data  Leakage
    scaler = StandardScaler()

    # calculate means and standard division and do it on train data
    X_train_scaled = scaler.fit_transform(X_train)

    # make scaling on X_test
    X_test_scaled = scaler.transform(X_test)

    # arr to df
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }

def descriptive():
    return 0


get_data()

