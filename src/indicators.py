def add_indicators(df):

    """ technical indicators are extra calcs that help analyse trends and patterns in stock prices
    - useful for prediction models and trading analysis

    below, [] is used to access cols in a df
    rolling(), mean() is a built in pandas method
    """

    # moving averages 
    df["MA_10"] = df["Close"].rolling(10).mean()    # 10 day moving average. creates a sliding 10-day window (looks at 10 rows at a time - e.g days 1-10, 2-11)
    df["MA_50"] = df["Close"].rolling(50).mean()

    # daily return
    df["Daily_Return"] = df["Close"].pct_change()   # pct = percentage change. calcs percentage incr/decr between rows. formula used: new_price - old_price/old_price

    # prediction target 
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)    # 1 = stock goes up. 0 = stock goes down (created for a ml classification model)

    # .shift(-1) = moves col up by 1 (compares tomorrows and todays closing)
    # astype converts boolean vals to ints (1, 0) - from [True, False] to [1, 0]

    # volume
    df["Volume_MA_10"] = df["Volume"].rolling(10).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_10"]      # todays vol vs average - detects spikes

    # volatility
    df["Volatility"] = df["Daily_Return"].rolling(10).std()     # how much price fluctuates

    # momentum
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)       # price change over last 5 days 
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    # price distance from moving averages
    df["Dist_MA_10"] = (df["Close"] - df["MA_10"]) / df["MA_10"]    # how far price is from MA_10
    df["Dist_MA_50"] = (df["Close"] - df["MA_50"]) / df["MA_50"]

    return df   # returns modified df with new indicator col added

def calc_rsi(df, window=14):

    """
    measure whether a stock is overbought (price may fall soon) or oversold (price may rise soon)
    RSI > 70 = overbought
    RSI < 30 = oversold 
    """

    delta = df["Close"].diff()              # delta = change in value. value changes between consecutive closing prices (output: 5, -2). pos values -> stock went up

    gain = delta.where(delta > 0, 0)        # extracting gains by keeps pos changes, neg values become 0
    loss = -delta.where(delta < 0, 0)       # extracting losses by converting losses into pos values. - cancels out - value to turn into pos (treated as pos magnitudes)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi             # new df col 

    return df 

def calc_macd(df):      

    """
    moving avg convergence divergence
    identifies trends, momentum, possible buy/sell signals
    compares two moving averages 

    MACD > signal line = bullish trend
    MACD > signal line = bearish trend 
    """

    ema12 = df["Close"].ewm(span=12).mean()         # ema = exponential moving avg (ema gives more importance to recent prices). ewm = exponential weighted moving 
    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26                  # 12 day reacts quickly to price changes, 26 days react more slowly (difference helps identify trend direction and momentum)

    df["Signal_Line"] = df["MACD"].ewm(span=9).mean()       # smooths MACD line to make trend signals easier to see 

    return df 

def create_trade_signals(df):

    """
    generate buy or sell signals based on indicators
    """

    df["Buy_Signal"] = (
        (df["MACD"] > df["Signal_Line"]) &
        (df["RSI"] < 70)
    )

    df["Sell_Signal"] = (
        (df["MACD"] < df["Signal_Line"]) &
        (df["RSI"] > 30)
    )

    return df

def calc_bollinger_bands(df, window=20):
    """
    measure volatility around ma
    price near upper band = potentially overbought
    price near lower band = potentially oversold 
    """
    df["BB_Middle"] = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()                     # std of closing price
    df["BB_Upper"] = df["BB_Middle"] + (2*std)                  # 2 std above middle 
    df["BB_Lower"] = df["BB_Middle"] - (2*std)

    return df

# if __name__ == "__main__":
