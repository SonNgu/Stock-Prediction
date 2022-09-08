# File: stok_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import statistics
import os
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from parameters import *

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
DATA_SOURCE = "yahoo"
COMPANY = "TSLA"

# Number of days to look back to base the prediction
PREDICTION_DAYS = 100 # Original
LOOKUP_STEPS = 1

TRAIN_START = dt.datetime(2012, 5, 23)     # Start date to read
TRAIN_END = dt.datetime(2020, 1, 7)       # End date to read
def load_data(test_size = 0.2, lookup_step = 1, split_by_date = True, shuffle = True, scale = True):
    if isinstance(COMPANY, str):
        # load it from yahoo_fin library
        data = pd.DataFrame(web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END)) # Read data using yahoo
    elif isinstance(COMPANY, pd.DataFrame):
        # already loaded, use it directly
        data = COMPANY
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    if "date" not in data.columns:
        data["date"] = data.index

    data.dropna(inplace=True)

    scale_data = data.copy()

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in ["Open","Close","High","Low","Adj Close"]:
            scaler = preprocessing.MinMaxScaler()
            scale_data[column] = scaler.fit_transform(np.expand_dims(scale_data[column].values, axis=1))

    data['future'] = data['Adj Close'].shift(-lookup_step)

    sequence_data = []
    sequences = deque(maxlen=PREDICTION_DAYS)

    for entry, target in zip(data[["Open","Close","High","Low","Adj Close"] + ["date"]].values, data['future'].values):
        sequences.append(entry)
        if len(sequences) == PREDICTION_DAYS:
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
    else:    
        # split the dataset randomly
        x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=test_size, shuffle=shuffle)
    x_train = x_train[:, :, :len(["Open","Close","High","Low","Adj Close"])].astype(np.float32)
    x_test = x_test[:, :, :len(["Open","Close","High","Low","Adj Close"])].astype(np.float32)

    return data, x_train, y_train, x_test, y_test, scale_data
# It could be a bug with pandas_datareader.DataReader() but it
# does read also the date before the start date. Thus, you'll see that 
# it includes the date 22/05/2012 in data!
# For more details: 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------

# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here

# Flatten and normalise the data
# First, we reshape a 1D array(n) to 2D array(n,1)
# We have to do that because sklearn.preprocessing.fit_transform()
# requires a 2D array
# Here n == len(scaled_data)
# Then, we scale the whole array to the range (0,1)
# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
# values.reshape(-1, 1) 
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# When reshaping an array, the new shape must contain the same number of elements 
# as the old shape, meaning the products of the two shapes' dimensions must be equal. 
# When using a -1, the dimension corresponding to the -1 will be the product of 
# the dimensions of the original array divided by the product of the dimensions 
# given to reshape so as to maintain the same number of elements.

# To store the training data
data, x_train, y_train, x_test, y_test, scale_data = load_data()

if not os.path.isdir("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/results"):
    os.mkdir("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/results")

if not os.path.isdir("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/logs"):
    os.mkdir("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/logs")

if not os.path.isdir("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/data"):
    os.mkdir("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/data")

date_now = time.strftime("%Y-%m-%d")
filename = os.path.join("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/data", f"{COMPANY}_{date_now}.csv")

data.to_csv(filename)

import plotly.graph_objects as go
import plotly.express as px

window_size = 1
def plot_graph(window_size=window_size):
    data_mean = data.rolling(window_size).mean()
    data_mean = data_mean.iloc[window_size-1 :: window_size, :]
    trace1 = {
        'x': data_mean.index,
        'open': data_mean["Open"],
        'close': data_mean["Close"],
        'high': data_mean["High"],
        'low': data_mean["Low"],
        'type': 'candlestick',
        'name': COMPANY,
        'showlegend': True
    }
    fig = go.Figure(data=[trace1])
    fig.update_traces(hovertext = 'trading days: ' + str(window_size))
    open = pd.DataFrame({'value':data_mean['Open'],'column':'open'})
    close = pd.DataFrame({'value':data_mean['Close'],'column':'close'})
    high = pd.DataFrame({'value':data_mean['High'],'column':'high'})
    low = pd.DataFrame({'value':data_mean['Low'],'column':'low'})
    adjclose = pd.DataFrame({'value':data_mean['Adj Close'],'column':'adjclose'})
    box_data = pd.concat([open, close, high, low, adjclose])
    fig2 = px.box(box_data, x = 'column', y = 'value', title = COMPANY + " " + str(window_size) + " consecutive trading day/s")
    fig.show()
    fig2.show()

plot_graph()

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

model_name = f"{date_now}_{COMPANY}-\{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{PREDICTION_DAYS}-step-{LOOKUP_STEPS}-layers-{N_LAYERS}-units-{UNITS}"

model = create_model(PREDICTION_DAYS, len(["Open","Close","High","Low","Adj Close"]), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

checkpointer = ModelCheckpoint(os.path.join("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/logs", model_name))

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)


