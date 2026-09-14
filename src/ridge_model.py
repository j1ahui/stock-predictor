from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler                        # scaler class that standardises with a standard deviation of 1 (-1 to 1)
from sklearn.pipeline import make_pipeline

import numpy as np 
import pandas as pd 
import joblib

FEATURES = [                        # input features (col names). gives model multiple indicators describing current state of stock
    "Close",
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

def train_ridge(df: pd.DataFrame, target_column: str, model_path: str) -> tuple[Ridge, np.ndarray, dict[str, float]]:
    """
    Train a Ridge regression model and evaluate its predictions.
    """
    # df = df[FEATURES + [target_column]].dropna().copy()

    X = df[FEATURES]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))                  # alpha controls how strongly Ridge regression penalises large coefficients (regular linear regression tries to minimise prediction error while ridge minimises both prediction error and adds a penalty)
    print("Ridge dataframe:", df.shape)
    print("Ridge X:", X.shape)
    print("Ridge y:", y.shape)
    
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


