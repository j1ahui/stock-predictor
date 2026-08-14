# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

# creating a stock market web app 
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_stock_dataset
from src.indicators import add_indicators, calc_rsi, calc_macd, create_trade_signals, calc_bollinger_bands
from src.model import train_model
from src.lstm_model import train_lstm, predict_next, prepare_data
from src.backtest import generate_signals_rf, generate_signals_lstm, backtest, sharpe_ratio, max_drawdown, calc_risk_score, walk_forward, monte_carlo


from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import streamlit as st      # user interface
import yfinance as yf 
import plotly.graph_objects as go
import joblib, os
import pandas as pd
import tensorflow as tf
import numpy as np

# ----------------- CSS -----------------

with open("app/styles.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html = True
    )

# ----------------- PAGE CONGIF -----------------
st.set_page_config(page_title = "AI Stock Prediction Dashboard", layout="wide")

# ticker = st.text_input("Ticker", "AAPL")    # creates an input box. Ticker parameter = label shown to user. "AAPL" = default parameter 

#@st.cache_resource
#def cached_lstm(df):                    # wrapper function (adds extra functionality - caching)

#    return train_lstm(df)

@st.cache_resource
def load_lstm_model():
    model = load_model("models/lstm_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

# ----------------- LANDING PAGE TITLE -----------------
st.title("PREDICT TOMORROW")

st.markdown("## Market Overview")

# ----------------- TOP PERFORMING STOCKS -----------------

stocks = ["AAPL", "GOOG", "MSFT", "NVDA", "TSLA"]

table_data = []

for stock in stocks:
    
    data = yf.download(stock, period="5d")

    current_price = data["Close"].iloc[-1].squeeze()
    previous_price = data["Close"].iloc[-2].squeeze()

    perct_change = (current_price - previous_price) / previous_price * 100

    table_data.append({
        "Stock": stock,
        "Price": round(float(current_price), 2),
        "% Change": perct_change

    })

market_df = pd.DataFrame(table_data)                    # creating a new dataframe object 

st.dataframe(market_df, use_container_width=True)

# ----------------- SELECT -----------------

st.markdown("## Stock")

col1, col2 = st.columns(2)

with col1:

    ticker = st.selectbox("Choose", ["AAPL", "TSLA", "NVDA", "MSFT", "GOOG"])



with col2: 
    model_type = st.selectbox(
        "Model",
        ["Random Forest", "LSTM"],
        key = "model_type"
    )

# ----------------- SESSION STATE -----------------

if "selected_page" not in st.session_state:

    st.session_state.selected_page = "Charts"

if ticker and model_type:

    st.session_state.selected_page = "Predictions"

# ----------------- AUTO NAV -----------------

if "selected" not in st.session_state:

    st.session_state["selected"] = False 

if ticker and model_type:

    st.session_state["selected"] = True

# ----------------- LOAD DATA  -----------------

df = load_stock_dataset(ticker)
df = add_indicators(df)
df = calc_rsi(df)
df = calc_macd(df)
df = calc_bollinger_bands(df)
df = create_trade_signals(df)

# data = yf.download(ticker, period="1y")

# ----------------- BOTTOM ICON NAV  -----------------

selected = option_menu(

    menu_title = None,
    options = [
        "Charts",
        "Indicators",
        "Predictions"
    ],
    icons = ["graph-up", "bar-chart", "robot"],
    default_index = ["Charts", "Indicators", "Predictions"].index(st.session_state.selected_page),
    orientation = "horizontal"
)


# ----------------- tab1  -----------------

if selected == "Charts":

    st.subheader(ticker, "Stock Charts")

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

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Upper"],
            name="BB Upper",
            line=dict(color="rgba(173, 204, 255, 0.8)", dash="dot"),
      
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Lower"],
            name="BB Lower",
            line=dict(color="rgba(173, 204, 255, 0.8)", dash="dot"),        
            fill="tonexty",                          # shades the area between upper and lower bands
            fillcolor="rgba(173, 204, 255, 0.08)"
        )
    )

    st.plotly_chart(fig, width="stretch")

# ----------------- tab2  -----------------

elif selected == "Indicators":

    st.subheader("Technical Indicators")

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

    st.subheader("Bollinger Bands")
    latest = df.iloc[-1]
    st.write(df[["Close", "BB_Upper", "BB_Lower"]].tail())

    if latest["Close"] > latest["BB_Upper"]:
        st.warning("Price above upper band - potentially overbought")
    elif latest["Close"] < latest["BB_Lower"]:
        st.info("Price below lower band - potentially oversold")
    else:
        st.success("Price within bands - normal range")
        
# ----------------- tab3  -----------------

elif selected == "Predictions": 

    st.subheader("Predictions")

    current_price = df["Close"].iloc[-1]

    st.write("Current Price: ", round(float(current_price), 2))

    if model_type == "Random Forest":
        model_path = "models/random_forest.pkl"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            _, accuracy, X_test, predictions, probabilities = train_model(df)

        else:
            model, accuracy, X_test, predictions, probabilities = train_model(df)
            
        # df["Target"] = (
        #     df["Close"].shift(-1) > df["Close"]
        # ).astype(int)

        # model, accuracy = train_model(df)

        # current_price = df["Close"].iloc[-1]

        st.subheader("Model Predictions - Random Forest")

        st.write("Current Price", round(current_price, 2))

        prediction = "UP" if accuracy > 0.5 else "DOWN" 
        st.write("Prediction: ", prediction)
        st.write("Model Accuracy; ", round(accuracy * 100, 2), "%")

        # ----------------- ACTUAL VS PREDICTED GRAPH  -----------------

        df_test = df.loc[X_test.index].copy()         # creates a copy of test rows 
        df_test["Predicted"] = predictions

        df_test["Predicted_Price"] = df_test["Close"].where(            # creates col based on condition
            df_test["Predicted"] == 1,                # use this line if condition true
            df_test["Close"] * 0.99                   # else use this 
        )

        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Scatter(
                x=df_test.index,
                y=df_test["Close"],
                name="Actual Price",
                line=dict(color="rgba(100, 200, 100, 0.9)")
            ))
        
        fig_pred.add_trace(
            go.Scatter(
                x=df_test.index,
                y=df_test["Predicted_Price"],
                name="Predicted Price",
                line=dict(color="rgba(255, 165, 0, 0.5)")
            ))
        
        fig_pred.update_layout(
            title="Actual vs Predicted - Random Forest",
            xaxis_title = "Date",
            yaxis_title = "Price"
        )

        st.plotly_chart(fig_pred, width="stretch")


        # ----------------- BACKTEST  -----------------


        signals = generate_signals_rf(predictions, probabilities, threshold=0.6)

        final_capital, pnl_history = backtest(
            df.loc[X_test.index].reset_index(drop=True),
            signals
        )

        st.subheader("Backtest Results")
        st.write("Starting Capital: $10,000")
        st.write("Final Capital: $", round(final_capital), 2)
        st.write("Sharpe Ratio", round(sharpe_ratio(pnl_history, rf=0.0), 4))
        st.write("Max Drawdown: $", round(max_drawdown(pnl_history), 2))

        risk_score = calc_risk_score(pnl_history)
        st.subheader("Risk Score")

        if risk_score <= 3:
            st.success(f"Risk Score: {risk_score} / 10 - Low Risk")
        elif risk_score <= 6:
            st.warning(f"Risk Score: {risk_score} / 10 - Medium Risk")
        else:
            st.error(f"Risk Score: {risk_score} / 10 - High Risk")

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            y=pnl_history,
            name="Portfolio Value",
            line=dict(color="rgba(100, 200, 100, 0.9)") 
        ))

        fig_pnl.update_layout(
            title="Portfolio Value Over Time - Random Forest",
            xaxis_title = "Days",
            yaxis_title = "Capital"
    
        )
        st.plotly_chart(fig_pnl, width="stretch")


        # ----------------- WALK FORWARD TESTING -----------------


        st.subheader("Walk Forward Testing")
        fold_results, wf_pnl = walk_forward(df)

        import pandas as pd 
        st.dataframe(pd.DataFrame(fold_results), width="stretch")

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            y = wf_pnl,
            name = "Walk Forward Portfolio",
            line = dict(color="rgba(100, 200, 255, 0.9)")
        ))

        fig_wf.update_layout(
            title = "Walk Forward Portfolio Value",
            xaxis_title = "Days",
            yaxis_title = "Capital ($)"

        )
        st.plotly_chart(fig_wf, width="stretch")


        # ----------------- MONTE CARLO -----------------


        st.subheader("Monte Carlo Simulation")

        simulations = monte_carlo(pnl_history)

        fig_mc = go.Figure() 

        for i, sim in enumerate(simulations):
            fig_mc.add_trace(go.Scatter(
                y = sim,
                mode = "lines",
                line = dict(width=0.5, color="rgba(100, 200, 255, 0.1)"),
                showlegend=False
            ))
        
        # overlay percentile lines 
        fig_mc.add_trace(go.Scatter(
            y = np.percentile(simulations, 95, axis = 0),
            name = "95th Percentile (Best)",
            line = dict(color="rgba(100, 200, 100, 0.9)")
        ))
        fig_mc.add_trace(go.Scatter(
            y = np.percentile(simulations, 50, axis = 0),
            name = "50th Percentile (Median)",
            line = dict(color="rgba(255, 255, 255, 0.9)")
        ))
        fig_mc.add_trace(go.Scatter(
            y = np.percentile(simulations, 5, axis = 0),
            name = "5th Percentile (Worst)",
            line = dict(color="rgba(255, 100, 100, 0.9)")
        ))

        fig_mc.update_layout(
            title = "Monte Carlo Simulation - 200 Scenarios (252 Days)",
            xaxis_title = "Days",
            yaxis_title = "Portfolio Value ($)"
        )

        st.plotly_chart(fig_mc, width="stretch")

        st.write("Best Case (95th): $", round(np.percentile(simulations[:, -1], 95), 2))
        st.write("Median Case (50th): $", round(np.percentile(simulations[:, -1], 50), 2))
        st.write("Worst Case (5th): $", round(np.percentile(simulations[:, -1], 5),2))




    # ------------------------

    elif model_type == "LSTM":

        st.subheader("Model Prediction - LSTM")


        import tensorflow as tf

        # tf.keras.backend.clear_session()
        model, scaler = load_lstm_model()

        # st.write("model loadeddddd")

        df_clean = df.dropna().copy()

        # st.write("BEFORE PREDICT")
        prediction = predict_next(model, df_clean, scaler)         # model, df, scaler are required as parameters for function arguments 
        # st.write("AFTER PREDICT")
        st.write("Predicted Price:", prediction)        

        current_price = df["Close"].iloc[-1]
        st.write("Current Price: ", round(current_price, 2))
        st.write("Predicted Next Close Price: ", round(float(prediction), 2))

        
        difference = prediction - current_price
        percentage_change = (difference / current_price) * 100
        st.write("Predicted Change: ", round(float(difference), 2))
        

        percentage_change = (difference / current_price) * 100
        st.write("Percentage Change:", round(float(percentage_change), 2), "%")

        # ----------------- ACTUAL VS PREDICTED GRAPH  -----------------

        X_lstm, y_lstm, scaler_check = prepare_data(df_clean)
        X_lstm = X_lstm.astype("float32")        
        lstm_preds = model(X_lstm, training=False).numpy()
        lstm_preds = scaler.inverse_transform(lstm_preds)
        y_actual = scaler.inverse_transform(y_lstm.reshape(-1, 1))          # changes into 2d cols from 1 (inverse_transform expects 2d array). reshape = (rows, cols) format 
        plot_index = df_clean.index[60:]

        fig_lstm = go.Figure()

        fig_lstm.add_trace(
            go.Scatter(
                x=plot_index,
                y=y_actual.flatten(),
                name="Actual Price",
                line=dict(color="rgba(100, 200, 100, 0.9)")
            ))
        
        fig_lstm.add_trace(
            go.Scatter(
                x=plot_index,
                y=lstm_preds.flatten(),
                name="Predicted Price",
                line=dict(color="rgba(255, 165, 0, 0.5)")
            ))
        
        fig_lstm.update_layout(
            title="Actual vs Predicted - LSTM",
            xaxis_title = "Date",
            yaxis_title = "Price"
        )

        st.plotly_chart(fig_lstm, width="stretch")

        # ----------------- BACKTEST  -----------------

        signals = generate_signals_lstm(lstm_preds.flatten(), y_actual.flatten())

        final_capital, pnl_history = backtest(
            df_clean.iloc[60:].reset_index(drop=True),
            signals
        )

        st.subheader("Backtest Results")
        st.write("Starting Capital: $10,000")
        st.write("Final Capital: ", round(float(final_capital), 2))
        st.write("Sharpe Ratio: ", round(sharpe_ratio(pnl_history), 4))
        st.write("Max Drawdown: $", round(max_drawdown(pnl_history), 2))

        risk_score = calc_risk_score(pnl_history)
        st.subheader("Risk Score")

        if risk_score <= 3:
            st.success(f"Risk Score: {risk_score} / 10 - Low Risk")
        elif risk_score <= 6:
            st.warning(f"Risk Score: {risk_score} / 10 - Medium Risk")
        else:
            st.error(f"Risk Score: {risk_score} / 10 - High Risk")


        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            y=pnl_history,
            name="Portfolio Value",
            line=dict(color="rgba(100, 200, 100, 0.9)")
        ))

        fig_pnl.update_layout(
            title="Portfolio Value Over Time - LSTM",
            xaxis_title = "Days",
            yaxis_title = "Capital ($)"
        )

        st.plotly_chart(fig_pnl, width="stretch")



# python3 -m streamlit run app/dashboard.py
# PYTHONPATH=. streamlit run app/dashboard.py
