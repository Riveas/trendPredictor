# Trend predictor
This project is a simple and fun app that allowed me to test posibilities of machine learning. Do not treat it like a investing application but rather tool that could be useful for predicting time based trends.
## Required modules:
* Scikit-learn
* Keras
* Pandas
* Numpy
* Matplotlib
* Yfinance
## Installation guide:
To ensure that your project will work fine first you'll need to install necessary modules. You can do it simply by running following commands in your terminal:  
pip install scikit-learn  
pip install keras  
pip install pandas  
pip install numpy  
pip install matplotlib  
pip install yfinance  
## Setup
First thing to do is importing data that would further be used for training and testing your model. (Remeber that testing data should differ from training data!):
```
apl = yft.Ticker("AAPL")
apl = apl.history("max")

train = apl.loc['2015-1-1':'2020-1-1'].copy()

test = apl.loc['2020-1-2':].copy()

total = pd.concat((train['Close'], test['Close']), axis = 0)
```
Then you'll want to scale your data so teaching process takes shorter time:
```
scaler = MinMaxScaler(feature_range = (0, 1))
aplScaled = scaler.fit_transform(train['Close'].values.reshape(-1, 1))
```
Next you'll want to set how many days you want to look in the past and initialize training arrays:
```
back = 30

xTrain = []
yTrain = []
```
Next you'll want to build your create arrays containing data you want to use for prediction and containing expected values:
```
for x in range (back, len(aplScaled)):
    xTrain.append(aplScaled[x - back:x, 0])
    yTrain.append(aplScaled[x, 0])

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
```
With such prepared data you are ready for creating your machine learning model:
```
model = Sequential()
model.add(LSTM(units = 200, return_sequences = True, input_shape = (back, 1)))
model.add(LSTM(units = 50))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mse')
model.fit(xTrain, yTrain, epochs = 20, batch_size = 25)
```
Next you'll have to prepare data that will be used for testing your model:
```
modelInput = total[len(total) - len(test) - back:].values
modelInput = modelInput.reshape(-1, 1)
modelInputScaled = scaler.transform(modelInput)
```
Then you'll have to prepare test data similarly to training data but without expected outcome:
```
for x in range (back, len(modelInputScaled)):
    xTest.append(modelInputScaled[x - back:x, 0])

xTest = np.array(xTest)
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
```
With everything prepared you can run your model and inverse transform data so it isn't between 0 and 1:
```
predicted = model.predict(xTest)
predicted = scaler.inverse_transform(predicted)
```
Lastly you can plot your predicted data and compare it to real values:
```
plt.plot(test['Close'].values, color = 'red', label = 'actual price')
plt.plot(predicted, color = 'green', label = 'predicted price')
plt.show()
```
As you can see below, data predicted by model looks fairly similar to actual data:
![trendComparison](https://user-images.githubusercontent.com/130605144/234022256-e3de88d0-790a-4959-86f6-42ed5026c38f.png)

