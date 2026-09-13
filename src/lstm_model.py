# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

import tensorflow as tf
# tf.config.run_functions_eagerly(True)       # exec mode (eager vs graph)

import numpy as np 
import pandas as pd
import os 
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping

WINDOW_SIZE = 60                                               # how much past info model looks at. each prediction gets 60 days of history. creates new window each day you move forward (overlaps)
FIVE_DAY_HORIZON = 5
NEXT_DAY_HORIZON = 1


FEATURES = [                        # input features (col names). gives model multiple indicators describing current state of stock
    "Close",
    "MA_10",
    "MA_50",
    "Daily_Return",
    # "Volume_Ratio",
    # "Volatility",
    # "Momentum_5",
    # "Momentum_10",
    # "Dist_MA_10",
    # "Dist_MA_50",
    # "RSI",
    # "MACD",
]


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, window_size=WINDOW_SIZE, horizon: int = 1):
    """
    Prepare stock price data allowing lstm to learn from previous days to predict following day or horizon (how far into future / prediction distance)

    Splits data into training and test splits.
    Scales training data and creates windows of input and target data.
    """

    data = df[FEATURES].dropna()

    print("DATA LENGTH:", len(data))
    print("DATA SHAPE:", data.shape)

    split_idx = int(len(data) * (1 - test_size))

    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    feature_scaler = MinMaxScaler()                             # creates object. scaler converts features into a range between 0 and 1. purpose is to make values smaller and consistent so ml can learn patterns more effectively 
    feature_scaler.fit(train_data)                              
    # scaled = scaler.fit_transform(data)                       # learns min and max values and then transforms !!!!! SCALER BEFORE SPLIT LEAK - scales min/max ceiling on all data, including test data.
   
    train_features_scaled = feature_scaler.transform(train_data)
    test_feature_with_history = pd.concat([train_data.iloc[-window_size:], test_data])
    test_features_scaled = feature_scaler.transform(test_feature_with_history)          # creates first window for first prediction. iloc[-window_size:] = last 60 rows

    target_scaler = MinMaxScaler()
    train_close = train_data[["Close"]]
    target_scaler.fit(train_close)

    train_target_scaled = target_scaler.transform(train_close)
    test_close_with_history = pd.concat([train_data[["Close"]].iloc[-window_size:], test_data[["Close"]]])
    test_target_scaled = target_scaler.transform(test_close_with_history)


    def make_windows(feature_data, target_data):                           # produces correct 3d shape instead of needing reshape
        """
        
        """
        X, y = [], []                                   # x = previous 60 days of prices (input data), y = next days price (target values)

        for i in range(window_size, len(feature_data) - horizon):       
            X.append(feature_data[i-window_size:i, ])        # NumPy slicing syntax. general slicing format is array[rows, cols] aka start:stop. i-window_size:i means from i-window_size to i. col 0 is the "Close" col 
            y.append(target_data[i + horizon - 1, 0])
            # X: [day 2, day 3, day 4]
            # y:  day 5
        return np.array(X), np.array(y)                 # converts python lists into numpy arrays 

    X_train, y_train = make_windows(train_features_scaled, train_target_scaled)
    X_test, y_test = make_windows(test_features_scaled, test_target_scaled)

    print("train_scaled:", train_features_scaled.shape)
    print("test_scaled:", test_features_scaled.shape)            # adds 60 prev days to make first prediction
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))      # [0] = samples, [1] = time steps (prev 60 days), final 1 = features (e.g "Close"). lstm layers require 3 dimensions (samples, time steps, features). reshape adds another dimension (features)
    # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print(f"ANSWER!!!!! {X_train.shape}, {y_train.shape}")

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler


def build_lstm(window_size: int = WINDOW_SIZE):
    """
    Build a LSTM model for next day stock prediction.

    Creates a two layer LSTM network followed by a dense output layer.

    Args: 
        window_size (int): number of previous days 
    Returns: 
        Sequential object: a compiled Keras LSTM model
    """
    model = Sequential()                                                    # sequential model object creation
    model.add(Input(shape=(window_size, len(FEATURES))))                    # every timestep has len(features) = 12
    model.add(LSTM(128, return_sequences=True))                              # adding layers to object (add is a method). layer 1 adds 50 LSTM units (neurons), pass full seq to next lstm layer (must stack), 60 timestamps (days) and 1 feature (close price)
    print("first lstm added")
    model.add(LSTM(50))  
    print("second")                                                           # layer 2. adds another 50 lstm units. outputs final learned representation
    model.add(Dense(1))                                                             # dense = fully connected neural network layer. output layer. adds 1 output neuron. this predicts one val (next stock price)
    print("dense added")
    model.compile(optimizer ="adam", loss="mean_squared_error")                     # adam = optimisation algo. "mean_squared_error" = measures prediction error
    print("3")
    return model


def train_lstm(df: pd.DataFrame, horizon: int = FIVE_DAY_HORIZON):
    """
    Train and save LSTM model.

    Returns:
        tuple: trained model, fitted scaler, test input data, test, target values
    """
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = prepare_data(df, horizon=horizon)
    model = build_lstm()

    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)                                                  # val_loss detects overfitting. patience=5 means training will stop after validation loss has failed to improve for 5 consecutive epochs. weights = number inside neural networks that the model learns (learning = algorithm is repeatedly adjusting its weights to reduce loss)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=False, verbose=1)          # epoch = one complete pass through the entire training dataset (model learns a little more each epoch). batch_size = groups of 32 at a time 
    
    save(model, feature_scaler, target_scaler)

    return model, feature_scaler, target_scaler, X_test, y_test


def predict_price(model: Sequential, df: pd.DataFrame, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler, window_size: int = WINDOW_SIZE, horizon: int = 1) -> float:
    """
    
    """
    data = df[FEATURES].iloc[-window_size:]                         # .values converts to numpy array (by extracting raw numerical array) from pandas df 
    scaled = feature_scaler.transform(data).astype("float32")

    X = scaled.reshape((1, window_size, len(FEATURES))).astype("float32")

    prediction = model(X, training = False).numpy()
    prediction = target_scaler.inverse_transform(prediction)               # inverse_transform() converts scaled values back to og scale
    prediction = prediction[0, 0]

    return float(prediction)


def prediction_evaluation(model, X_test, y_test, target_scaler):
    predictions = model.predict(X_test, verbose=0)                          # verbose = terminal progress output

    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    predictions_unscaled = target_scaler.inverse_transform(predictions)

    evaluation = evaluate_errors(y_test_unscaled, predictions_unscaled)

    print("Actual min: ", y_test_unscaled.min())
    print("Actual max: ", y_test_unscaled.max())

    print("Prediction min: ", predictions_unscaled.min())
    print("Prediction max: ", predictions_unscaled.max())

    return predictions_unscaled, evaluation


def evaluate_errors(y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    evaluation = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return evaluation


def save(model: Model, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler) -> None:

    os.makedirs("models", exist_ok = True)
    model.save("models/lstm_model.keras")
    joblib.dump(feature_scaler, "models/feature_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")

    print("Model + scalers saved successfully")
