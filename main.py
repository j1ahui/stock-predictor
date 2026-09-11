import faulthandler, signal
faulthandler.register(signal.SIGUSR1)

import joblib
import pandas as pd

from src.data_loader import load_stock_dataset
from src.indicators import add_indicators, calc_rsi, calc_macd, create_trade_signals, calc_bollinger_bands
from src.model import train_model
from src.lstm_model import prepare_data, train_lstm, predict_next, evaluate_errors, predict_5day
from src.regression_model import train_regression
from src.linear_regression import train_linear


def prepare_stock_data(ticker: str) -> pd.DataFrame:              # "AAPL", "TSLA"
    """
    Load and prep stock data for model training.
    """
    df = load_stock_dataset(ticker)

    df = add_indicators(df)
    df = calc_rsi(df)
    df = calc_macd(df)
    df = calc_bollinger_bands(df)
    df = create_trade_signals(df)

    return df 


"""
print(df.columns.tolist())
print(df.head().to_string())            # pandas function displaying first few rows of DataFrame (commonly used to quickly inspect data and check that it loaded correctly)
print(df.iloc[50:55].to_string())
print(df.head(4))    # pandas function displaying first few rows of DataFrame (commonly used to quickly inspect data and check that it loaded correctly)

print("MAIN DATA LENGTH:", len(df))
print("MAIN DATA SHAPE:", df.shape)

print("MAIN X:", X_lstm.shape)
print("MAIN y:", y_lstm.shape)

"""

# df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)


def run_random_forest(df: pd.DataFrame) -> dict:
    """
    Train Random Forest model and generate predictions
    """
    model_1day, accuracy_1day, X_test_1day, predictions_1day, probabilities_1day, X_latest = train_model(df, "Target_1Day", "models/random_forest_1day.pkl")            # returns a RandomForestClassifier object (model_1day)
    model_5day, accuracy_5day, X_test_5day, predictions_5day, probabilities_5day, _ = train_model(df, "Target_5Day", "models/random_forest_5day.pkl")

    prediction_1day = model_1day.predict(X_latest)[0]                   # indexing array returned by model
    probability_1day = model_1day.predict_proba(X_latest)[0, 1]         # returns a 2d array. [0, 1] = first row, second col

    prediction_5day = model_5day.predict(X_latest)[0]
    probability_5day = model_5day.predict_proba(X_latest)[0, 1]

    return {
        "model_1day": model_1day,
        "model_5day": model_5day,
        "accuracy_1day": accuracy_1day,
        "accuracy_5day": accuracy_5day,
        "prediction_1day": prediction_1day,
        "prediction_5day": prediction_5day,
        "probability_1day": probability_1day,
        "probability_5day": probability_5day,
    }


def run_lstm(df: pd.DataFrame) -> dict:
    """
    Train LSTM model and generate next-day prediction.
    """
    model, scaler, X_test, y_test = train_lstm(df)
    prediction, evaluation = predict_5day(model, X_test, y_test, scaler)

    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "prediction": prediction,
        "evaluation": evaluation,
    }


def run_regression(df: pd.DataFrame) -> dict:
    """
    Train regression model and predict price in 5 days.
    """
    model, predictions, metrics = train_regression(df, "Target_5Day_Price", "models/random_forest_regression_5day.pkl")

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics
    }


def run_linear(df: pd.DataFrame) -> dict:
    """
    Train linear regression model and predict price in 5 days.
    """
    model, predictions, metrics = train_linear(df, "Target_5Day_Price", "models/linear_regression_5day.pkl")

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics
    }


def main():
    ticker = "AAPL"

    df = prepare_stock_data(ticker)

    rf_result = run_random_forest(df)
    lstm_result = run_lstm(df)
    rf_regression_result = run_regression(df)
    linear_regression = run_linear(df)

    print("Stock: ", ticker)
    print("Random Forest 1-Day", rf_result["prediction_1day"])
    print("Random Forest 5-day: ", rf_result["prediction_5day"])
    print("LSTM: ", lstm_result["prediction"])
    print("Random Forest Regression: ", rf_regression_result["predictions"])
    print("Linear Regression: ", linear_regression["predictions"])


if __name__ == "__main__":
    main()

