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
PREDICTION_DAYS = 50 # Original
LOOKUP_STEPS = 15

TRAIN_START = dt.datetime(2012, 5, 23)     # Start date to read
TRAIN_END = dt.datetime(2020, 1, 7)       # End date to read
def load_data(test_size = 0.2, lookup_step = LOOKUP_STEPS, split_by_date = True, shuffle = True, scale = True):
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

    scaled_data = {}

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in ["Open","Close","High","Low","Adj Close"]:
            scaler = preprocessing.MinMaxScaler()
            scaled_data[column] = scaler.fit_transform(np.expand_dims(data[column].values, axis=1))
            column_scaler[column] = scaler

        scaled_data["column_scaler"] = column_scaler

    data['future'] = data['Adj Close'].shift(-lookup_step)

    data.dropna(inplace=True)

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

    dates = x_test[:, -1, -1]
    test_data = data.loc[dates]
    test_data = test_data[~test_data.index.duplicated(keep='first')]
    x_train = x_train[:, :, :len(["Open","Close","High","Low","Adj Close"])].astype(np.float32)
    x_test = x_test[:, :, :len(["Open","Close","High","Low","Adj Close"])].astype(np.float32)



    return data, x_train, y_train, x_test, y_test, scaled_data, test_data

data, x_train, y_train, x_test, y_test, scaled_data, test_data = load_data()

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

#plot_graph()

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

model_name = f"{date_now}_{COMPANY}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{PREDICTION_DAYS}-step-{LOOKUP_STEPS}-layers-{N_LAYERS}-units-{UNITS}"

model = create_model(PREDICTION_DAYS, len(["Open","Close","High","Low","Adj Close"]), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

checkpointer = ModelCheckpoint(os.path.join("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/results/", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/logs/", model_name))

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

def get_final_df(model, data):
    y_pred = model.predict(x_test)
    if SCALE:
        y_test_data = np.squeeze(scaled_data["column_scaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(scaled_data["column_scaler"]["Adj Close"].inverse_transform(y_pred))
    test_data[f"true_adjclose_{LOOKUP_STEPS}"] = y_test_data
    test_data[f"adjclose_{LOOKUP_STEPS}"] = y_pred
    
    data.sort_index(inplace=True)
    final_df = test_data
    
    return final_df


def forecast(model):
    prediction = model.predict(x_test[:LOOKUP_STEPS])
    dates = pd.to_datetime(data["date"])
    forecast_datetime = pd.date_range(list(dates)[-1], periods = LOOKUP_STEPS, freq='1d').tolist()
    forecast_dates = []
    for i in forecast_datetime:
        forecast_dates.append(i.date())

    if SCALE:
        predicted_price = scaled_data["column_scaler"]["Adj Close"].inverse_transform(prediction)
    else:
        predicted_price = prediction

    pred_df = pd.DataFrame(index = np.array(forecast_dates), columns = {'Adj Close':predicted_price})
    pred_df['Adj Close'] = predicted_price

    return pred_df

def plot_pred_graph(test_df, pred_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEPS}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEPS}'], c='r')
    plt.plot(pred_df, c='g')
    plt.plot()
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price","Forecast"])
    plt.show()

test_df = get_final_df(model, data)
pred_df = forecast(model)
plot_pred_graph(test_df, pred_df)
