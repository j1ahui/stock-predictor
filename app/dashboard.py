import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# creating a stock market web app 
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
from src.data_loader import load_stock_dataset
from src.indicators import add_indicators, calc_rsi, calc_macd, create_trade_signals
from src.model import train_model
from src.lstm_model import train_lstm, predict_next
from tensorflow.keras.models import load_model

import streamlit as st      # user interface
import yfinance as yf 
import plotly.graph_objects as go
import joblib



# ticker = st.text_input("Ticker", "AAPL")    # creates an input box. Ticker parameter = label shown to user. "AAPL" = default parameter 

#@st.cache_resource
#def cached_lstm(df):                    # wrapper function (adds extra functionality - caching)

#    return train_lstm(df)

# @st.cache_resource
def load_lstm_model():
    model = load_model("models/lstm_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

st.title("AI Stock Prediction Dashboard")

ticker = st.selectbox("Choose", ["AAPL", "TSLA", "NVDA", "MSFT", "GOOG"])

model_type = st.selectbox(
    "Model",
    ["Random Forest", "LSTM"],
    key = "model_type"
)

st.write("DEBUG MODEL TYPE:", model_type)

df = load_stock_dataset(ticker)
df = add_indicators(df)
df = calc_rsi(df)
df = calc_macd(df)
st.write("siganlsssss")
df = create_trade_signals(df)

# data = yf.download(ticker, period="1y")

st.line_chart(df["Close"])

fig = go.Figure(data=[          # creates chart container (main object that holds everything related to graph - think slideshow and its contents) data stores graph data 
    go.Candlestick(             # contents of chart container 
        x=df.index,             # dates/timestamps on x axis
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )
])

fig.update_layout(
    title=f"{ticker} Stock Price",
    xaxis_title="Date",
    yaxis_title = "Price"
)

st.write(df.head())

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA_10"],
        name = "MA 10"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA_50"],
        name = "MA 50"
    )
)

st.plotly_chart(fig)

st.write(df[[           # double sq brackets to select multiple cols 
    "Close",
    "RSI",
    "MACD",
    "Signal_Line"
]].tail())

buy_signals = df[df["Buy_Signal"]]
sell_signal = df[df["Sell_Signal"]]

st.subheader("Buy Signals")
st.write(buy_signals[["Close", "RSI", "MACD"]].tail())      # stores rows where buy signals happen

st.subheader("Sell Signals")
st.write(sell_signal[["Close", "RSI", "MACD"]].tail())

st.subheader("Latest RSI")
st.write(df["RSI"].iloc[-1])        # accesses rows by their numerical pos. -1 means last pos 
st.line_chart(df["RSI"])

st.subheader("MACD")
st.write(df[["MACD", "Signal_Line"]])

if df["MACD"].iloc[-1] > df["Signal_Line"].iloc[-1]:

    st.success("Bullish Signal")

else:

    st.error("Bearish Signal")

if model_type == "Random Forest":

    df["Target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    model, accuracy = train_model(df)

    st.subheader("Random Forest Results")

    st.write("Model Accuracy; ", round(accuracy, 2))

elif model_type == "LSTM":

    st.write("we in")

    st.subheader("LSTM Forecast")

    import tensorflow as tf

    tf.keras.backend.clear_session()
    model, scaler = load_lstm_model()

    st.write("model loadeddddd")

    st.write("DF LENGTH:", len(df))

        #prediction = predict_next(                      # model, df, scaler are required as parameters for function arguments 
        #    model,
         #   df,
          #  scaler
        #)


    df = df.dropna()

        #st.write("df shape: ", df.shape)

        #data = df[["Close"]].values[-60:]
        #st.write("🔥 RAW DATA SHAPE:", data.shape)

        # scaled = scaler.transform(data)
        # st.write("🔥 SCALED SHAPE:", scaled.shape)

        # X = scaled.reshape(1, 60, 1)
        # st.write("🔥 FINAL INPUT SHAPE:", X.shape)

        # st.write("ABOUT TO PREDICT")

    st.write("BEFORE PREDICT")
    prediction = predict_next(model, df, scaler)
    st.write("AFTER PREDICT")
    st.write(prediction)        

        #prediction = model(X, training=False).numpy()
        #prediction = prediction[0, 0]

        #prediction = scaler.inverse_transform([[prediction]])[0, 0]



    # current_price = df["Close"].iloc[-1]
    # st.write("Current Price: ", round(current_price, 2))
    # st.write("Predicted Next Close Price: ", round(prediction, 2))

    # difference = prediction - current_price
    # st.write("Predicted Change: ", round(difference, 2))
    # st.write("DEBUG prediction:", prediction)

tab1, tab2, tab3 = st.tabs([
    "Charts",
    "Indicators",
    "Predictions"
])

# python3 -m streamlit run app/dashboard.py
# PYTHONPATH=. streamlit run app/dashboard.py
