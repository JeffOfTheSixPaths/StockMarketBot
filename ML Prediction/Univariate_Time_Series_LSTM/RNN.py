#python
from gettext import npgettext
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
from keras.layers import LSTM


train = pd.read_csv('MSFT n= 7 m= 3.csv', usecols=['value 0','value 1','value 2','value 3','value 4','value 5','value 6']).to_numpy() # training data 
#train = np.expand_dims(train, 2)
test = pd.read_csv('AAPL n= 7 m= 3.csv', usecols=['value 0','value 1','value 2','value 3','value 4','value 5','value 6']).to_numpy() # testing data
#test = np.expand_dims(test, 2)
train_y = pd.read_csv('MSFT n= 7 m= 3.csv', usecols = ['value 0']).to_numpy() # testing data
test_y = pd.read_csv('AAPL n= 7 m= 3.csv', usecols = ['value 0']).to_numpy() # testing data
np.delete(train_y, 0)
np.delete(test_y, 0)
np.delete(train, len(train)-1)
np.delete(test, len(test)-1)
print(test_y)


model = models.Sequential()
#model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(100, activation='relu'))
model.add(LSTM(32, activation='relu', input_shape=(7,1), return_sequences=True))
model.add(keras.layers.Dense(1, activation="relu"))

adamOpti = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer=adamOpti,
              loss='mse',)

model.fit(train, train_y, epochs=200, validation_data=(test, test_y), shuffle=True)
model.save()
def predict(data_sequence):
    three_day_avg = np.array([])
    for i in range(3):
        np.append((three_day_avg, model.predict(data_sequence[-7:])), axis=None)
    return np.average(three_day_avg)

test_loss, test_acc = model.evaluate(test, test_y, verbose=1)
 
