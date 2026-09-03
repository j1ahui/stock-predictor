import joblib
import pandas as pd

from src.data_loader import load_stock_dataset
from src.indicators import add_indicators, calc_rsi, calc_macd, create_trade_signals, calc_bollinger_bands
from src.model import train_model
from src.lstm_model import prepare_data, train_lstm, predict_next


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
"""

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)


def run_random_forest(df: )

model_1day, accuracy_1day, X_test_1day, predictions_1day, probabilities_1day, X_latest = train_model(df, "Target_1Day", "models/random_forest_1day.pkl")            # returns a RandomForestClassifier object (model_1day)
model_5day, accuracy_5day, X_test_5day, predictions_5day, probabilities_5day, _ = train_model(df, "Target_5Day", "models/random_forest_5day.pkl")

prediction_1day = model_1day.predict(X_latest)[0]                   # indexing array returned by model
probability_1day = model_1day.predict_proba(X_latest)[0, 1]         # returns a 2d array. [0, 1] = first row, second col

prediction_5day = model_5day.predict(X_latest)[0]
probability_5day = model_5day.predict_proba(X_latest)[0, 1]

print("MAIN DATA LENGTH:", len(df))
print("MAIN DATA SHAPE:", df.shape)

X_lstm, y_lstm, X_test, y_test, scaler = prepare_data(df)

print("MAIN X:", X_lstm.shape)
print("MAIN y:", y_lstm.shape)
