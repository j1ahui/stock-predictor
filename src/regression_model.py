from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

def train_regression(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> tuple[RandomForestRegressor, np.ndarray, dict[str, float]]:
    """
    Train a Random Forest regression model and evaluate its predictions.

    Model learns relationship between training features and target values then inferences values for test samples.
    """
    model = RandomForestRegressor(n_estimators=200, random_state=33)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return model, predictions, {
        "MAE": mae,
        "RMSE": rmse,
        "r2": r2
    }