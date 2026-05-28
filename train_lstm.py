import tensorflow as tf
tf.config.run_functions_eagerly(True)

import os
import joblib
import numpy as np
import yfinance as yf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler


# ----------------------------
# 1. DATA PREP
# ----------------------------
def prepare_data(df):

    data = df[["Close"]].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    window_size = 60

    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


# ----------------------------
# 2. MODEL
# ----------------------------
def build_lstm():

    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# ----------------------------
# 3. TRAIN
# ----------------------------
def train():

    print("Downloading data...")

    df = yf.download("AAPL", period="2y")

    print("Preparing data...")

    X, y, scaler = prepare_data(df)

    print("Building model...")

    model = build_lstm()

    print("Training model...")

    model.fit(
        X, y,
        epochs=3,
        batch_size=32,
        verbose=1
    )

    return model, scaler


# ----------------------------
# 4. SAVE MODEL
# ----------------------------
def save(model, scaler):

    os.makedirs("models", exist_ok=True)

    model.save("models/lstm_model.h5")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model saved!")


# ----------------------------
# 5. MAIN
# ----------------------------
if __name__ == "__main__":

    model, scaler = train()
    save(model, scaler)