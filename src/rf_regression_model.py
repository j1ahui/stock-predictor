from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV

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

def train_regression(df: pd.DataFrame, target_column: str, model_path: str) -> tuple[RandomForestRegressor, np.ndarray, dict[str, float]]:
    """
    Train a Random Forest regression model and evaluate its predictions.

    Model learns relationship between training features and target values then inferences values for test samples.
    """
    df = df[FEATURES + [target_column]].dropna().copy()             # df = df.dropna().copy()

    X = df[FEATURES]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    tscv = TimeSeriesSplit(n_splits=5)                                  # n_splits=5 means create 5 chronological validation splits while respecting time order

    param_grid = {                                                      # keys are rf parameter names
        "n_estimators": [100, 200, 300, 500],                           # no. trees
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    }

    search = RandomizedSearchCV(                                    # search/tuning object that manages rf models
        estimator=RandomForestRegressor(random_state=10),           # estimator = ml model to tune
        param_distributions=param_grid,                             # combos to try
        n_iter=20,                                                  # randomly try 20 different combinations of those parameters (4 × 4 × 3 × 3 × 3 = 432 combinations)
        scoring="neg_root_mean_squared_error",
        cv=tscv,                                                    # use tscv (time series cross validation) object for cross validation aka 5 chronological validation splits
        random_state=10,
        n_jobs=-1                                                   # how many CPU cores to use (-1 = all available CPU cores)
    )

    # model = RandomForestRegressor(n_estimators=200, random_state=10)

    search.fit(X_train, y_train)
    model = search.best_estimator_                                      # best_estimator_ = attribute of search object. best rf found
    predictions = model.predict(X_test)

    print("Best parameters:", search.best_params_)
    print("Best CV RMSE:", -search.best_score_)                         # scikit learns search functions are designed so higher scores are better (-10 > -20)

    mae = mean_absolute_error(y_test, predictions)                      # calculates average absolute error (uses absolute values)
    rmse = mean_squared_error(y_test, predictions) ** 0.5               # calcs average error (squares errors first) but sensitive to large differences
    r2 = r2_score(y_test, predictions)                                  # displays variation between actual and predicted prices (0-1 scale where 1 conveys perfect predictions. 0.8 = explains ~80% of variation)

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

    print("Training target range:", y_train.min(), "to", y_train.max())
    print("Test target range:", y_test.min(), "to", y_test.max())
    print("Prediction range:", predictions.min(), "to", predictions.max())

    joblib.dump(model_data, model_path)


    return model, predictions, evaluation