import pandas as pd
import datetime as DT
import numpy as np
import matplotlib.pyplot as plt
from binance.enums import *
from binance.client import Client
from collections import deque
import json
import time




cfg = ''
# read json file and store data into cfg var
# make sure that you provide responsible data

with open("config.json","r") as config:
    cfg = json.load(config)

# geting data from cfg
api_key = cfg['api_key']

api_secret = cfg['api_secret']

symbol = cfg['symbol']
client = Client(api_key, api_secret)

def update_api_keys():
    """
    Updates the API keys in the config file and global variables
    """
    global api_key, api_secret

    file_path = "config.json"
    
    # Prompt the user for input
    api_key = input("Enter the API key: ")
    api_secret = input("Enter the API secret key: ")
    
    # Load the existing data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Update the API key and secret key in the data dictionary
    data["api_key"] = api_key
    data["api_secret"] = api_secret
    
    with open(file_path, 'w') as file:
        json.dump(data, file)

def stop_Loss():
    """
    Calculates the stop loss value based on historical data
    
    @return: stop loss value
    @param: float
    """
    
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=6)
    week_ago = week_ago.strftime('%d %b, %Y')
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, str(week_ago))
    highVal = [float(entry[2]) for entry in klines]
    lowVal = [float(entry[3]) for entry in klines]
    closeVal = [float(entry[4]) for entry in klines]
    avgDownDrop = (sum(highVal)/len(highVal)-sum(lowVal)/len(lowVal))/(sum(closeVal)/len(closeVal))
    stopVal = closeVal[-2]*(1-avgDownDrop)
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
    sell |= df['RSI'] > 70
    if (df['close'].iloc[-1] < stop_Loss()):
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

    rsi = compute_RSI(data['close'])   
    data['RSI'] = rsi
    macd = compute_MACD()
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

def compute_MACD():
    """
    Computes the Moving Average Convergence Divergence (MACD) for the given data.
    @return: MACD indicator string value.

    """
    today = DT.date.today()
    week_ago = today - DT.timedelta(days=6)
    week_ago = week_ago.strftime('%d %b, %Y')
    
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, str(week_ago))
    print(type(klines))
    print("1 ", klines)
    
    closeVal = [float(entry[4]) for entry in klines]
    closeVal = pd.DataFrame(closeVal)
    print(type(closeVal))
    print(closeVal)
    ema12 = closeVal.ewm(span=12).mean()
    ema26 = closeVal.ewm(span=26).mean()
    macd = ema26 - ema12
    signal = macd.ewm(span=9).mean()
    macd = macd.values.tolist()
    signal = signal.values.tolist()
    if macd[-1] > signal[-1] and macd[-2] < signal[-2]:
        macdIndicator = 'BUY'
    elif macd[-1] < signal[-1] and macd[-2] > signal[-2]:
        macdIndicator = 'SELL'
    else:
        macdIndicator = 'HOLD'
    
    return macdIndicator


def download_data(interval, start_str):
    """
    Downloads historical data for the specified interval and start date
    
    @param interval: time interval for data
    @type interval: str
    @param start_str: start date as string
    @type start_str: str
    @return: downloaded data
    """
    klines = client.get_historical_klines(symbol, interval, start_str)

    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'asset_volume', 'numb_of_trades', 'buy_base_asset', 'buy_asset_volume', 'ignore'])
    
    columns_to_delete = ['asset_volume', 'numb_of_trades', 'buy_base_asset', 'buy_asset_volume', 'ignore']
    data = data.drop(columns_to_delete, axis=1)
    
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['close'] = data['close'].astype(float)
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


# def plot_results(data,que_time, que_price,rangNumb):
    """
    Plots the results based on the given data
    
    @param data: backtesting results
    @type data: pandas series type
    @type que_time: int 
    @type que_price: int
    """

    if rangNumb:
        que_time.append(list(data.iloc[:,0])[ len(data.iloc[:,0])-200: -1])
        que_price.append(list(data.iloc[:,4])[ len(data.iloc[:,4])-200: -1])
    if not rangNumb:
        que_time.append(list(data.iloc[:,0])[-1])
        que_price.append(list(data.iloc[:,4])[-1])

    plt.plot(que_time,que_price)
    plt.scatter(que_time, que_price)

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Trading Results for {symbol}')

    plt.draw()
    plt.pause(60)
    plt.clf()


def test_trading():
    """
    Performs test trading on historical data
    
    """
    
    # que_price = deque(maxlen = 200)
    # que_time = deque(maxlen=200)

    rangNumb = True
    while True:
        data = download_data('1m', '1 Jun 2023')
        
        data = backtest(data)
        print(data)
        # plot_results(data, que_time, que_price, rangNumb)
        rangNumb = False
         
        

def live_trading():
    client = Client(api_key, api_secret)
    prev_buy_signal = False
    prev_sell_signal = False
    print('_________TIME________SYMBOL_SIGNAL__PRICE_______RSI__________MACD__')
    while True:
        data = download_data( '1m', '1 Jun 2023')
        calculate_indicators(data)
        buy_signal, sell_signal = strategy(data)
        buy_signal = buy_signal.iloc[-1]
        sell_signal = sell_signal.iloc[-1]
        if buy_signal and not prev_buy_signal:
            print(f" {symbol} {data['close'].iloc[-1]} {data['RSI'].iloc[-1]}  {data['MACD'].iloc[-1]}")
            order = client.create_test_order(
                symbol = symbol,
                side='BUY',
                type='MARKET',
                quantity=0.0005,
            )
        elif sell_signal and not prev_sell_signal:
            print(f"{symbol} SELL: {data['close'].iloc[-1]} {data['RSI'].iloc[-1]} {data['MACD'].iloc[-1]}")
            order = client.create_test_order(
                symbol = symbol,
                side='SELL',
                type='MARKET',
                quantity=0.0005,
            )
        else:
            print(f"{data['timestamp'].iloc[-1]} {symbol} HOLD: {data['close'].iloc[-1]} {data['RSI'].iloc[-1]} {data['MACD'].iloc[-1]}")

        prev_buy_signal = buy_signal
        prev_sell_signal = sell_signal
        time.sleep(60)
    
def launch_bot():
    """
    Takes user input and performs different actions based on the input value
    """
    flag = False
    print("To launch Trading Bot choose the mode: \n 1. Back test: \n 2. Online trading.")
    while not flag:
        user_input = int(input("Press 1 or 2: "))
        if user_input == 1:
            test_trading()
            flag = True
        elif user_input == 2:
            update_api_keys()
            live_trading()
            flag = True
        else:
            print("Invalid input.")
            
launch_bot()