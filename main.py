import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import yfinance as yft
from sklearn.preprocessing import MinMaxScaler

apl = yft.Ticker("AAPL")
apl = apl.history("max")

train = apl.loc['2015-1-1':'2020-1-1'].copy()

test = apl.loc['2020-1-2':].copy()

scaler = MinMaxScaler(feature_range = (0, 1))
aplScaled = scaler.fit_transform(train['Close'].values.reshape(-1, 1))

tail = 30

xTrain = []
yTrain = []

for x in range (tail, len(aplScaled)):
    xTrain.append(aplScaled[x-tail:x, 0])
    yTrain.append(aplScaled[x, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (tail, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(xTrain, yTrain, epochs = 50, batch_size = 25)

print(test['Close'].values)

total = pd.concat((train['Close'], test['Close']), axis = 0)

modelInput = total[len(total)-len(test)-tail:].values
modelInput = modelInput.reshape(-1, 1)
modelInputScaled = scaler.transform(modelInput)

xTest = []

for x in range (tail, len(modelInputScaled)):
    xTest.append(modelInputScaled[x-tail:x, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

predicted = model.predict(xTest)
predicted = scaler.inverse_transform(predicted)

print(predicted)

plt.plot(test['Close'].values, color = 'red', label = 'actual price')
plt.plot(predicted, color = 'green', label = 'predicted price')
plt.show()