import os
import time
from tensorflow.keras.layers import LSTM, GRU

N_LAYERS = 2
# LSTM cell
CELL = GRU
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters
scale = True

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 25
EPOCHS = 1