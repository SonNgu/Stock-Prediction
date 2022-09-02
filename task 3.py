import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import statistics

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

COMPANY = "TSLA"

def load_data(ticker = COMPANY, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):

    if isinstance(ticker, str):
        # load it from yahoo_fin library
        data = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        data = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
        
    if "date" not in data.columns:
        data["date"] = data.index

    data['future'] = data['adjclose'].shift(-lookup_step)

    last_sequence = np.array(data[feature_columns].tail(lookup_step))

    data.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(data[feature_columns + ["date"]].values, data['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    
    if split_by_date:
        train_samples = int((1 - test_size) * len(X))
        x_train = X[:train_samples]
        y_train = y[:train_samples]
        x_test  = X[train_samples:]
        y_test  = y[train_samples:]

    return data, x_train, y_train, x_test, y_test
    
data, x_train, y_train, x_test, y_test = load_data()

import plotly.graph_objs as go
import plotly.express as px
window_size = 3
def plot_graph(window_size=window_size):
    data_mean = data.rolling(window_size).mean()
    data_mean = data_mean.iloc[window_size-1 :: window_size, :]
    trace1 = {
        'x': data_mean.index,
        'open': data_mean["open"],
        'close': data_mean["close"],
        'high': data_mean["high"],
        'low': data_mean["low"],
        'type': 'candlestick',
        'name': COMPANY,
        'showlegend': True,
    }
    fig = go.Figure(data=[trace1])
    fig.update_traces(hovertext = 'trading days: ' + str(window_size))
    open = pd.DataFrame({'value':data_mean['open'],'column':'open'})
    close = pd.DataFrame({'value':data_mean['close'],'column':'close'})
    high = pd.DataFrame({'value':data_mean['high'],'column':'high'})
    low = pd.DataFrame({'value':data_mean['low'],'column':'low'})
    adjclose = pd.DataFrame({'value':data_mean['adjclose'],'column':'adjclose'})
    box_data = pd.concat([open, close, high, low, adjclose])
    fig2 = px.box(box_data, x = 'column', y = 'value', title = COMPANY + " " + str(window_size) + " consecutive trading day/s")
    fig.show()
    fig2.show()

plot_graph()
