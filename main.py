from src.data_loader import load_stock_dataset
from src.indicators import add_indicators, calc_rsi, calc_macd, create_trade_signals, calc_bollinger_bands
from src.model import train_model
from src.lstm_model import prepare_data, build_lstm, train_lstm, predict_next

df = load_stock_dataset("AAPL")
df = add_indicators(df)

print(df.columns.tolist())
print(df.head().to_string())    # pandas function displaying first few rows of DataFrame (commonly used to quickly inspect data and check that it loaded correctly)
print(df.iloc[50:55].to_string())

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

model_1day, accuracy_1day, X_test_1day, predictions_1day, probabilities_1day, X_latest = train_model(df, "Target_1Day")
model_5day, accuracy_5day, X_test_5day, predictions_5day, probabilities_5day, _ = train_model(df, "Target_5Day")

prediction_1day = model_1day.predict(X_latest)[0]                   # indexing array returned by model
probability_1day = model_1day.predict_proba(X_latest)[0, 1]         # returns a 2d array. [0, 1] = first row, second col

prediction_5day = model_5day.predict(X_latest)[0]
probability_5day = model_5day.predict_proba(X_latest)[0, 1]


print(df.head(4))    # pandas function displaying first few rows of DataFrame (commonly used to quickly inspect data and check that it loaded correctly)

df = calc_rsi(df)

df = calc_macd(df)

df = calc_bollinger_bands(df)

df = create_trade_signals(df)

df = prepare_data(df)

df = build_lstm()

# df = train_lstm(model, df, scaler)

# df = predict_next(model, df, scaler)

print(df)


# can add a value to empty parentheses to specify how many rows to display 