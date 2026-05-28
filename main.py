from src.data_loader import load_stock_dataset
from src.indicators import add_indicators, calc_rsi, calc_macd, create_trade_signals
from src.model import train_model
from src.lstm_model import prepare_data, build_lstm, train_lstm, predict_next

df = load_stock_dataset("AAPL")

df = add_indicators(df)

df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

model, accuracy = train_model(df)

print(df.head(4))    # pandas function displaying first few rows of DataFrame (commonly used to quickly inspect data and check that it loaded correctly)

print("Model Accuracy: ", accuracy)

df = calc_rsi(df)

df = calc_macd(df)

df = create_trade_signals(df)

df = prepare_data(df)

df = build_lstm()

# df = train_lstm(model, df, scaler)

# df = predict_next(model, df, scaler)

print(df)


# can add a value to empty parentheses to specify how many rows to display 