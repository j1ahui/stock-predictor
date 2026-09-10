from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd 
import joblib

FEATURES = [                        # input features (col names). gives model multiple indicators describing current state of stock
    "MA_10",
    "MA_50",
    "Daily_Return",
    "Volume_Ratio",
    "Volatility",
    "Momentum_5",
    "Momentum_10",
    "Dist_MA_10",
    "Dist_MA_50",
    "RSI",
    "MACD",
]

def train_regression(df: pd.DataFrame, target_column: str, model_path: str) -> tuple[RandomForestRegressor, np.ndarray, dict[str, float]]:
    """
    Train a Random Forest regression model and evaluate its predictions.

    Model learns relationship between training features and target values then inferences values for test samples.
    """
    df = df.dropna().copy()
    
    X = df[FEATURES]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=200, random_state=33)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    evaluation = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    model_data = {
        "model": model, 
        "predictions": predictions,
        "evaluation": evaluation
    }

    joblib.dump(model_data, model_path)


    return model, predictions, evaluation