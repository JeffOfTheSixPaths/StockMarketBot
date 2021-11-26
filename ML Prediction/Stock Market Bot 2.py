from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout, BatchNormalization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
 
#load data from csv
df=pd.read_csv("aaap.us.csv", usecols = ['Open'])
df = df.to_numpy()

scl = MinMaxScaler(feature_range=(0,1))
scl_data = scl.fit_transform(df.reshape(-1,1))


#preprocessing
predict_days = 50 #batch size

x_train, y_train = [], []

for x in range(predict_days, len(scl_data)):
    x_train.append(scl_data[x-predict_days:x, 0])
    y_train.append(scl_data[x, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))

