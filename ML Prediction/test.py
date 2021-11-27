from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout, BatchNormalization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

#parameters
split_percent = 0.8 #Parameter for splitting portions into test and train
lookback = 8 #The days to look back in supervised sequences

#Read data from csv; Data and Close columns required
df = pd.read_csv("aaap.us.csv", usecols = ['Close'])
df = df.to_numpy()
date = pd.read_csv("aaap.us.csv", usecols = ['Date'])
date = date.to_numpy()

predata = df.reshape((-1,1))

#data preprocessing to split data into test and train data
data_split = int(len(predata)*split_percent)
print(data_split)
train_data = predata[:data_split]
test_data = predata[data_split:]
train_date = date[:data_split]
test_date = date[data_split:]

#Modification from sequenced data to supervised data
supervised_train = TimeseriesGenerator(train_data, train_data, length=lookback, batch_size=20)
supervised_test = TimeseriesGenerator(test_data, test_data, length=lookback, batch_size=1)

#Create Model
model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(lookback,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 300
model.fit(supervised_train, epochs=num_epochs, validation_data=supervised_test, shuffle=False)
prediction = model.predict_generator(supervised_test)


predata = predata.reshape((-1))
train_data = train_data.reshape((-1))
test_data = test_data.reshape((-1))
prediction = prediction.reshape((-1))

plt.figure(figsize=(12,8))
plt.plot(prediction, label="Actual")
plt.plot(predata, label="Predicted")
plt.legend()
plt.title("Train Dataset")
plt.show()

#future predictions
