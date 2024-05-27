# IMPORT THE LIBRARY
import yfinance as yf
from datetime import datetime
import pandas as pd
import datetime 
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import pytz
import json

global symbol

def update_asset():
    global symbol
    file_path = "cnf.json"
    with open(file_path, 'r') as file:
        data = json.load(file)

    symbol = input("Enter asset: ")
    data["symbol"] = symbol

    with open(file_path, 'w') as file:
        json.dump(data, file)


def stop_Loss(symbol):
    """
    Calculates the stop loss value based on historical data
    
    @param symbol: Stock symbol or ticker
    @type symbol: str
    @return: Stop loss value
    @rtype: float
    """
    # Download historical data
    end_date = datetime.date.today()
    week_ago = end_date - datetime.timedelta(days=6)
    
    data = yf.download(symbol, start=week_ago, end=end_date, interval='1d')
    
    highVal = data['High']
    lowVal = data['Low']
    closeVal = data['Close']
    
    avgDownDrop = (highVal.mean() - lowVal.mean()) / closeVal.mean()
    stopVal = closeVal.iloc[-2] * (1 - avgDownDrop)
    
    return stopVal

def strategy(df):
    """ Returns the boolean value of buy and sell signal according to the conditions
    
    @:param df: historical data of symbol
    @:type pandas series type
    @:returns: boolean value of buy and sell
    """
    
    buy = df['RSI'] < 30
    buy |=df['MACD'] == 'BUY'
    sell = df['MACD'] == 'SELL'
    sell = df['RSI'] > 70
    if (df['Close'].iloc[-1] < stop_Loss(symbol)):
        df['stopLoss'] = 'ACTIVATE'
        sell = True
    df['stopLoss'] = 'SLEEP'
    return buy, sell

def calculate_indicators(data):
    """
    Calculates the technical indicators for the given data frame
    
    @param df: historical data of symbol
    @type df: pandas series type
    """

    rsi = compute_RSI(data['Close'])   
    data['RSI'] = rsi
    macd = compute_MACD(symbol)
    data['MACD'] = macd
    
def compute_RSI(data):
    """
    Computes the Relative Strength Index (RSI) for the given data
    
    @param data: closing prices
    @type data: pandas series type
    @param time_window: time window for RSI calculation
    @type time_window: int
    @return: RSI values
    """
    
    time_window=14
    
    diff = np.diff(data)
    up = 0 * diff
    down = 0 * diff

    up[diff > 0] = diff[ diff>0 ]
    down[diff < 0] = diff[ diff < 0 ]

    up = pd.DataFrame(up)
    down = pd.DataFrame(down)
    
    up_avg   = up.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_avg = down.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_avg/down_avg)
    rsi = 100 - 100/(1+rs)   
    rsi = pd.Series([np.nan] + rsi.squeeze().tolist(), index=data.index)

    return rsi


def compute_MACD(symbol):
    """
    Calculates the MACD (Moving Average Convergence Divergence) for the specified stock symbol and date range,
    and generates a buy, sell, or hold signal based on the MACD values.
    
    @param symbol: Stock symbol or ticker
    @type symbol: str
    @param start_date: Start date of the data range
    @type start_date: str
    @param end_date: End date of the data range
    @type end_date: str
    @return: String representing the buy, sell, or hold signal
    """
    # Download historical data
    end_date = datetime.date.today()
    week_ago = end_date - datetime.timedelta(days=6)
    
    data = yf.download(symbol, start=week_ago, end=end_date, interval='1d')

    # Calculate MACD using pandas' rolling mean function
    closeVal = data['Close']
    ema12 = closeVal.ewm(span=12).mean()
    ema26 = closeVal.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] < signal.iloc[-2]:
        macdIndicator = 'BUY'
    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] > signal.iloc[-2]:
        macdIndicator = 'SELL'
    else:
        macdIndicator = 'HOLD'
    
    return macdIndicator


def download_data():
    """
    Downloads historical data for the specified interval and start date
    
    @param interval: time interval for data
    @type interval: str
    @param start_str: start date as string
    @type start_str: str
    @return: downloaded data
    """
    # data = symbol.history(start_date, interval)#

    end_date = datetime.datetime.now(pytz.utc)
    start_date = end_date - datetime.timedelta(days=30)


    # Format dates as strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Download historical data
    data = yf.download(symbol, end=end_date_str,  start=start_date_str, interval='1h')

    del data['Adj Close']
    # del data['Stock Splits']
    del data['Volume']
    return data

def backtest(data):
    """
    Performs backtesting on the given data and returns the results
    
    @param data: historical data of symbol
    @type data: pandas series type
    @return: backtesting results
    """
    calculate_indicators(data)
    buy, sell = strategy(data)
    
    data['buy'] = buy
    data['sell'] = sell

    return data



def test_trading():
    """
    Performs test trading on historical data
    
    """

    while True:
        data = download_data()
        data = backtest(data)
        print(data)
        time.sleep(60)

def launch_bot():
    
    flag = False
    
    print("To launch Trading Bot choose the mode: \n 1. Back test: \n 2. Online trading.")
    while not flag:
        user_input = int(input("Press 1 or 2: "))
        if user_input == 1:
            
            update_asset()
            test_trading()
            flag = True
        elif user_input == 2:
            
            update_asset()
            test_trading()
            flag = True
        else:
            print("Invalid input.")
# PROMPT USER TO ENTER A NEW ASSET SYMBOL
launch_bot()


