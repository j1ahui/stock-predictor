import tensorflow as tf
# tf.config.run_functions_eagerly(True)       # exec mode (eager vs graph)

import numpy as np 
import os 
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras import Input

def prepare_data(df):

    """
    prepares stock price data so ml can learn from previous days to predict following days 
    """
    
    data = df[["Close"]].dropna()

    scaler = MinMaxScaler()                     # creates MinMaxScaler object. scaler converts vales into a range between 0 and 1. purpose is to make values smaller and consistent so ml can learn patterns more effectively 
    scaled = scaler.fit_transform(data)         # learns min and max values and then transforms 

    X, y = [], []           # x = previous 60 days of prices (input data), y = next days price (target values)

    window_size = 60        # last 60 days, predict next day 

    for i in range(window_size, len(scaled)):       
        X.append(scaled[i-window_size:i, 0])        # NumPy slicing syntax. general slicing format is array[rows, cols] aka start:stop. i-window_size:i means from i-window_size to i. col 0 is the "Close" col 
        y.append(scaled[i, 0])                      # adds next days value as target output 

    X, y = np.array(X), np.array(y)                 # converts python lists into numpy arrays 

    # lstm layers require 3 dimensions (samples, time steps, features). reshape adds another dimension (features)
    X = X.reshape((X.shape[0], X.shape[1], 1))      # X.shape returns the dimensions of the array
    # [0] = samples, [1] = time steps (prev 60 days), final 1 = features (e.g "Close")
    print(X.shape)
    print(y.shape)
    return X, y, scaler 

"""
def build_lstm():

    model = Sequential()

    model.add(Input(shape=(60, 1)))

    model.add(LSTM(4))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model

"""
def build_lstm():

    model = Sequential()        # layers are added one after another

    model.add(LSTM(50, return_sequences = True, input_shape = (60, 1)))     # passing arguments. layer 1. adds 50 LSTM units (neurons), pass full seq t next lstm layer (must stack), 60 timestamps (days) and 1 feature (close price)
    model.add(LSTM(50))                                                     # layer 2. adds another 50 lstm units. outputs final learned representation
    model.add(Dense(1))                                                     # dense = fully connected neural network layer. output layer. adds 1 output neuron. this predicts one val which is next stock price 

    model.compile(optimizer = "adam", loss = "mean_squared_error")          # adam = optimisation algo. "mean_squared_error" = measures prediction error

    return model

def train_lstm(df):

    print("prep stage")

    X, y, scaler = prepare_data(df)

    print("finsihed prep")

    print("buildinggg")

    model = build_lstm()

    print("starting fit")

    print(X.dtype)
    print(y.dtype)

    print(np.isnan(X).sum())
    print(np.isnan(y).sum())

    print(np.isinf(X).sum())
    print(np.isinf(y).sum())

    model.fit(X, y, epochs = 3, batch_size = 32, verbose = 1)       # epoch = one complete pass through the entire training dataset (model learns a little more each epoch). batch_size = groups of 32 at a time 

    print("finished fittt")

    return model, scaler 

'''
def predict_next(model, df, scaler):

    print("called")

    if len(df) < 60:
        return 0.0
    
    print("checked")

    data = df[["Close"]].values[-60:].astype("float32")          # .values converts to numpy array (by extracting raw numerical array) from pandas df 

    print("converted ")
    scaled = scaler.transform(data).astype("float32")
    print("sclaed")
    # X = np.array([scaled])                      # creates a new numpy array from existing array. also creates another array and adds extra dimension with []
    X = scaled.reshape((1, 60, 1)).astype("float32")

    print("reshaped")

    prediction = model.predict(X, verbose=0)

    print("predicted")

    prediction = scaler.inverse_transform(prediction)[0, 0]



    print("inversed")

    return float(prediction)
'''
def predict_next(model, df, scaler):

    print("STEP 1")

    data = df[["Close"]].values[-60:].astype("float32")

    print("STEP 2")

    scaled = scaler.transform(data).astype("float32")

    print("STEP 3")

    X = scaled.reshape((1, 60, 1)).astype("float32")

    print("STEP 4")

    # prediction = model(X, training = False).numpy()

    prediction = tf.convert_to_tensor(X)
    prediction = model(prediction, training=False)
    prediction = prediction.numpy()

    print("STEP 5")

    prediction = scaler.inverse_transform(prediction)

    print("STEP 6")

    prediction = prediction[0, 0]

    print("STEP 7")

    return float(prediction)

def save(model, scaler):

    os.makedirs("models", exist_ok = True)
    model.save("models/lstm_model.keras")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model + scaler saved successfully")

print(os.path.exists("models/lstm_model.keras"))
print(os.path.exists("models/scaler.pkl"))


if __name__ == "__main__":

    # model = build_lstm()
    # model.summary()

    import yfinance as yf

    df = yf.download("AAPL", period="2y")
    prepare_data(df)
    build_lstm()

    model, scaler = train_lstm(df)

    predict_next(model, df, scaler)

    save(model, scaler)


# source venv/bin/activate
