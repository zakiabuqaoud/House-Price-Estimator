
# California Housing Prediction App
Target => calculate Median House Value in Dollars by using API App


# *********  Results and Comparisons  **********

Model v1 (Linear Regression) -> MAE: 0.5332 | RMSE: 0.7456
Model v2 (Random Forest)    -> MAE: 0.3277 | RMSE: 0.5060
mae_v2 is 0.3277 and it is smaller than from mae_v1 0.5332
The error rate is lower in Random Forest Regression
Random Forest Regression is the best model for house price estimator


# *********  Models  **********

 1- model_v1: "Linear Regression (Linear Baseline)"
 2- model_v2: "Random Forest Regressor"

# *********  Project Structure  **********
1) app #API APP
=> main.py # this file is start point (start run from here to turn on API App)
=> routes.py # FastApi code and API Path
=> Schemas.py # classes shows parameter which attach it in body API and response form
2) src # Codes which train, build, evaluate models
=> get_data.py # load, Split and scaling data set that come from skilearn library
   (from here start run to build models)
=> train.py  # train two models and there are Linear regression and Random Forest Regression
=> evaluate.py # evaluate two models by calculate mae and rmse
3) models # here => model_v1.pkl (Linear regression) and (Random Forest Regression)  model_v2.pkl
4) .gitignore 
5) requirement.txt
6) .venv
7) readme.md

# *********  Features and Target Variable  **********
Features:
1) MedInc (Income Mediam)
2) HouseAge (house age)
3) AveRooms (number of rooms)
4) AveBedrms (number of bedroooms)
5) Population
6) AveOccup (number of residence)
7) Latitude
8) Longitude

Target 
1) House Price

# *********  Debugging Log  **********

1- High Prediction Error

trouble: When was model_v1 and using "/predict" =>  the returned house price predictions were unrealistically

The reason for the problem is that it must appling standerscaler then passing data input

2- Problem Solution 
Loaded scaler.pkl and updated the code in routes.py when be v1:
  if version == "v1":
      input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

