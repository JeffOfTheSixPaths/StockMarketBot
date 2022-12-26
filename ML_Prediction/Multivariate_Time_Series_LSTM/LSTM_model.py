#python
from gettext import npgettext #might be superfluous
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
from keras.layers import LSTM
#from sklearn.preprocessing import MinMaxScaler


#Preproccess data
def get_numpy(csv_name, label):
    '''
    Since arrays are stored inside the pandas dataframe cells, to_numpy() is not going to work well
    Instead, this function iterates through the cells, takes them as string items, then converts them into a numpy array
    '''
    df = pd.read_csv(csv_name, usecols=[
        label], sep='\t') #read csv with usecols parameter

    arr = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],dtype=float) #this is to make an array with shape (1,16) so that concatenation can occur

    for i in range(len(df)):
        
        ind = str(df.loc[i].item()) #take the object out of the cell and take as item
        
        #remove the brackets as np.fromstring does not work with brackets
        ind = ind.replace('[','')
        ind = ind.replace(']','')
        ind = ind.replace(', ', '|')

        arr = np.concatenate((arr, [np.fromstring(ind, dtype=float, sep='|')]), axis=0) #concatenate
    
    arr = np.delete(arr, 0, 0) #delete first index

    return arr

dataset = np.concatenate((get_numpy("IDIDIT.csv", "Lead Paragraph D1"), get_numpy("IDIDIT.csv", "Lead Paragraph D2")), axis=1)
print(dataset.shape)
print(get_numpy("IDIDIT.csv", "Lead Paragraph D1").shape)

def create_LSTM():
    inputs = layers.Input(shape=(356,7,8))
    layer1 = LSTM(128, input_shape=())(inputs)
    outputs = layers.Dense(1, activation="relu")(layer1)
    return keras.Model(inputs=inputs, outputs=outputs)