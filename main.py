import numpy as np
import pandas
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import yfinance as yft
from sklearn.preprocessing import MinMaxScaler

apl = yft.Ticker("AAPL")
apl = apl.history("max")

apl = apl.loc['2015-1-1':].copy()

scaler = MinMaxScaler(feature_range = (0, 1))
aplScaled = scaler.fit_transform(apl['Close'].values.reshape(-1, 1))

tail = 30

xTrain = []
yTrain = []

for x in range (tail, len(aplScaled)):
    xTrain.append(aplScaled[x-tail:x, 0])
    yTrain.append(aplScaled[x, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

#model = Sequential()
#model.add(LSTM(units = 50, return_sequences = True, input_shape = (tail, 1)))
#model.add(Dropout(0.2))
#model.add(LSTM(units = 50, return_sequences = True)
#model.add(Dropout(0.2))
#model.add(LSTM(units = 50, return_sequences = True)
#model.add(Dropout(0.2))
#model.add(Dense(units = 1))

#model.compile(optimizer = 'adam', loss = 'mse')
#model.fit(xTrain, yTrain, epochs = 10, batch_size = 32)

print(xTrain)
