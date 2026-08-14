import numpy as np 

def generate_signals_rf(predictions, probabilities=None, threshold=0.6):
    """
    threshold = minimum confidence to trigger a buy signal
    default 0.6 = model must be 60% confidence to buy
    if no probabilities passed, falls back to basic 0/1 predictions
    """
    if probabilities is not None:
        return(probabilities[:, 1] >= threshold).astype(int)        # numpy/pandas indexing 2d array. probabilities[:, 1] = take every row but only column no. 1

    return (predictions == 1).astype(int)

def generate_signals_lstm(pred_prices, actual_prices):

    return (pred_prices > actual_prices).astype(int)

def backtest(df, signals):

    initial_capital = 10000
    capital = initial_capital
    position = 0
    entry_price = 0
    pnl_history = []

    for i in range(len(signals)):                   # 1 = buy, 0 = sell

        price = df["Close"].iloc[i]

        if signals[i] == 1 and position == 0:
            position = 1
            entry_price = price

        elif signals[i] == 0 and position == 1:
            profit = price - entry_price
            capital += profit
            position = 0

        pnl_history.append(capital)

    return capital, np.array(pnl_history)

def sharpe_ratio(returns, rf=0.0, periods_per_year=252):                # rf = risk free return per free. ppy = 252 trading days                                   
    """
    how much return you get for each unit of risk (reward vs risk)
    high sharpe ratio = good returns for low risk
    low sharpe = weak returns / too much volatility 
    negative = strategy loses money on a risk adjusted basis
    """

    mean = np.mean(returns)
    std = np.std(returns)

    if std == 0:
        return 0

    # returns = np.diff(pnl_history) / pnl_history[:-1]           # simplified version of sharpe .diff = calcs diff between consecutive elements. -1 slicing = take everything except last

    return (mean * periods_per_year - rf) / (std * np.sqrt(periods_per_year))       # basically same as mean return / volatility 

def max_drawdown(pnl):
    """
    how far your portfolio has fallen from its highest value (peak) so far
    """
    peak = np.maximum.accumulate(pnl)               # stores max value so far. outputs [10000, 10500, 11000, 11000, 11000]
    drawdown = pnl - peak

    return drawdown.min()

def calc_volatilty(pnl_history, periods_per_year=252):
    """
    annualised volatility measures how much portfolio fluctuates 
    higher = more risk
    lower = more stable 
    """

    returns = np.diff(pnl_history) / pnl_history[:-1]
    daily_vol = np.std(returns)
    annualised_vol = daily_vol * np.sqrt(periods_per_year)

    return annualised_vol

def calc_risk_score(pnl_history, periods_per_year=252):
    """
    risk score from 1-10 based on volatility, drawdown and sharpe ratio
    1 = very low risk
    10 = very high risk
    """

    vol = calc_volatilty(pnl_history, periods_per_year=periods_per_year)
    drawdown = abs(max_drawdown(pnl_history)) / pnl_history[0]
    sharpe = sharpe_ratio(pnl_history, periods_per_year=periods_per_year)        # rf not required as it defaults to 0.0. P_P_P is also optional as it defaults to 252

    vol_score = min(vol * 20, 10)                               # calc vol * 20, but never let it exceed 10 (score is capped at 10). high vol = high risk 
    drawdown_score = min(drawdown * 20, 10)                     # high drawdown = high risk
    sharpe_score = max(0, 10 - (sharpe + 1) * 2)                # never lets it go below 0. low/negative = high risk

    score = (vol_score * 0.4) + (drawdown_score * 0.4) + (sharpe_score * 0.2)       # weighted avg

    return round(min(max(score, 1), 10), 1)


def walk_forward(df, n_splits=5):
    """
    Splits data into n_splits chunks.
    Trains on each chunk, tests on the next.
    Returns capital history across all folds.
    """

    # chunk = fold 

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    df = df.dropna()

    features = [
        "MA_10", 
        "MA_50", 
        "Daily_Return",
        "Volume_Ratio",
        "Volatility",
        "Momentum_5",
        "Momentum_10",
        "Dist_MA_10",
        "Dist_MA_50",
        "RSI",
        "MACD",
    ]

    X = df[features]                                        # can also write as X = df[["MA_10", "MA_50", "Daily_Return"]]
    Y = df["Target"]

    fold_size = len(df) // n_splits                         # returns num of rows. fold = one chunk of dataset. (suppose len(df) = 1000 then fold_size = 1000 // 5 = 200). 5 x 200 chunks = 5 folds but fold_size = 200 
    all_pnl = []
    fold_results = []

    for i in range(n_splits - 1):

        X_train = X.iloc[:fold_size * (i + 1)]                  # dataframe uses col names + index. df["Close"] selects col named "Closed", while df.iloc[1] selects the row at pos 1
        Y_train = Y.iloc[:fold_size * (i + 1)]
        X_test = X.iloc[fold_size * (i + 1) : fold_size * (i + 2)]
        Y_test = Y.iloc[fold_size * (i + 1) : fold_size * (i + 2)]

        model = RandomForestClassifier()
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(Y_test, predictions)
        signals = generate_signals_rf(predictions)

        df_fold = df.iloc[fold_size * (i + 1) : fold_size * (i + 2)].reset_index(drop=True)         # reset)index() renumbers index rows. drop=True discards old index 
        capital, pnl = backtest(df_fold, signals)

        all_pnl.extend(pnl)                             # appends elements individually, append() appends lists 
        fold_results.append({                           # appending a dict to fold_results list. python lists can contain ay type of objects 
            "Fold": i + 1,
            "Accuracy": round(accuracy * 100, 2),
            "Final Capital": round(capital, 2)
        })

    return fold_results, np.array(all_pnl)              # numpy arrays are optimized for mathematical calcs


def monte_carlo(pnl_history, n_simulations=200, n_days=252):
    """
    Runs n_simulations randomized simulations of portfolio performance.
    Based on historical daily returns from pnl_history.
    Returns array of shape (n_simulations, n_days)
    """

    returns = np.diff(pnl_history) / pnl_history[:-1]
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    simulations = []
    starting_value = pnl_history[-1]

    for _ in range(n_simulations):                                                  # use _ because we dont use iterator after loop
        daily_returns = np.random.normal(mean_return, std_return, n_days)           # normal() generates random numbers from a normal (Gaussian) distribution. its params are np.random.normal(mean, standard_deviation, size)
        prices = [starting_value]                           

        for r in daily_returns:
            prices.append(prices[-1] * (1 + r))             # prices[-1] = same index -> compounding returns. each simulation builds one possible future portfolio path based on randomly generated daily returns

        simulations.append(prices)

    return np.array(simulations)

