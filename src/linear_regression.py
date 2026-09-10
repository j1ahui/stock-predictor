from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

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

def train_linear(df: pd.DataFrame, target_column: str) -> tuple[LinearRegression, np.ndarray]:
    """
    Train a Linear regression model and evaluate its predictions.
    """
    df = df.dropna().copy()

    X = df[FEATURES]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)