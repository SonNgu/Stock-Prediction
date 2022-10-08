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
from statsmodels.tsa.arima_model import ARIMA

def load_data():
    model_1 = pd.DataFrame
    model_1.to_pickle("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/results/2022-10-08_TSLA-huber_loss-adam-LSTM-seq-50-step-15-layers-2-units-256.pkl")

    model_2 = pd.DataFrame
    model_2.to_pickle("C:/Users/SonyN/Desktop/BB-GAMCS/Y3S2/Intelligent Systems/Project B/Stock-Prediction/stock-prediction v0.1/results/2022-10-08_TSLA-huber_loss-adam-GRU-seq-50-step-15-layers-2-units-256.pkl")