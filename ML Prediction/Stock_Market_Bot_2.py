from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout, BatchNormalization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.preprocessing import MinMaxScaler
 
df=pd.read_csv("aaap.us.csv", usecols = ['Open'])
df = df.to_numpy()
scl = MinMaxScaler(feature_range=(0,1))
data_scaled = scl.fit_transform(df.reshape(-1,1))
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    print(np.array(X))
    return np.array(X),np.array(Y)
 
 
lb=5
X,y = processData(data_scaled,lb)
X_train,X_test = X[:int(X.shape[0]*0.90)],X[int(X.shape[0]*0.90):]
y_train,y_test = y[:int(y.shape[0]*0.90)],y[int(y.shape[0]*0.90):]
 
#Build the model
model = Sequential()
model.add(LSTM(256,input_shape=(lb,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
 
 
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
 
#Fit model with history to check for overfitting
history = model.fit(X_train,y_train,epochs=300,validation_data=(X_test,y_test),shuffle=False)

Xt = model.predict(X_train)
print(Xt)

plt.figure(figsize=(12,8))
plt.plot(scl.inverse_transform(y_train.reshape(-1,1)), label="Actual")
plt.plot(scl.inverse_transform(Xt), label="Predicted")
plt.legend()
plt.title("Train Dataset")
plt.show()

#future price
model_inputs = df[len(df) - len(X_test) - lb:]
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scl.fit_transform(model_inputs)

actual_data = [model_inputs[len(model_inputs) + 1 - lb:len(model_inputs) + 1,0]]
actual_data = np.array(actual_data)
actual_data = np.reshape(actual_data, (actual_data.shape[0], actual_data.shape[1], 1))

prediction = model.predict(actual_data)
prediction = scl.inverse_transform(prediction)
print(prediction)
