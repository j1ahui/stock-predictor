import yfinance as yf   # yfinance lib

def load_stock_dataset(ticker, period="2y"):    # ticker = stock symbol for wanted data. period = data length. example: load_stock_dataset("AAPL")
    stock = yf.Ticker(ticker)   # stock object. this lines creates an object connected to that stock 

    # print(dir(stock)) # displays attributes and methods for the object 

    # stock.    # dropdown shows available methods (methods are attached to objects)

    # help(stock.history)

    # df = data frame (a table structure from pandas library which is a python lib for data in tables). pandas help with storing, organising, cleaning, analysing data

    # yfinance creates DataFrame instead of having to manually use pandas lib
    df = stock.history(period=period)   # history method downloads stock price data. history() uses pandas internally to create a dataframe. can use dataframe methods because df is already a dataframe 

    return df 

""" 
yfinance - gets stock data
pandas - stores and analyses the data
"""

# Inside yfinance (simplified)
# def history(...):
#     import pandas as pd

#     data = ...
#     df = pd.DataFrame(data)
#     return df