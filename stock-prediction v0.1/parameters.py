import os
import time
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN

# Number of days to look back to base the prediction
PREDICTION_DAYS = 50 # Original
LOOKUP_STEPS = 15
COMPANY = "TSLA"

date_now = time.strftime("%Y-%m-%d")

N_LAYERS = 2
# LSTM cell
CELL = SimpleRNN
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters
SCALE = True

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 50


LOSS2 = "huber_loss"
OPTIMIZER2 = "adam"

N_LAYERS2 = 3

CELL2 = SimpleRNN
UNITS2 = 256
DROPOUT2 = 0.4

model1_name = f"{date_now}_{COMPANY}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{PREDICTION_DAYS}-step-{LOOKUP_STEPS}-layers-{N_LAYERS}-units-{UNITS}"
model2_name = f"{date_now}_{COMPANY}-{LOSS2}-{OPTIMIZER2}-{CELL2.__name__}-seq-{PREDICTION_DAYS}-step-{LOOKUP_STEPS}-layers-{N_LAYERS2}-units-{UNITS2}"