#python
from gettext import npgettext #might be superfluous
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense
#from keras.layers import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Normalizer
from Preprocessing_tools import NDNormalizer
import matplotlib.pyplot as plt

split_percent = 0.75

#Preproccess data
def get_numpy(csv_name, label, size):
    '''
    Since arrays are stored inside the pandas dataframe cells, to_numpy() is not going to work well
    Instead, this function iterates through the cells, takes them as string items, then converts them into a numpy array
    '''
    df = pd.read_csv(csv_name, usecols=[
        label], sep='\t') #read csv with usecols parameter

    arr = np.ones((size,),dtype=float) #this is to make an array with shape (1,16) so that concatenation can occur

    for i in range(len(df)):
        
        ind = str(df.loc[i].item()) #take the object out of the cell and take as item
        
        #remove the brackets as np.fromstring() does not work with brackets
        ind = ind.replace('[','')
        ind = ind.replace(']','')
        ind = ind.replace(', ', '|')
        
        arr = np.column_stack((arr, np.fromstring(ind, dtype=float, sep='|'))) #concatenate
    arr = arr.swapaxes(0,1)
    arr = np.delete(arr, 0, 0) #delete first index

    return arr

'''
def create_LSTM():
    inputs = layers.Input(shape=(355,7,8))
    layer1 = LSTM(128, input_shape=())(inputs)
    outputs = layers.Dense(1, activation="relu")(layer1)
    return keras.Model(inputs=inputs, outputs=outputs)
'''

'''
the keras.layers.LSTM() object takes inputs in the form of 3D tensor with shape [batch, timesteps, feature].
Thus, the features (statistics and various types of prices etc.) must be reshaped to be in the third axis/dimension
of the tensor, and append the timesteps (the days D1-D7) to the second axis/dimension
Firstly, the prices will be appended to the sentiment features in the features axis, which will now be in the second axis,
then, the tensor will be reshaped to be 3D, before the second and third axis is swapped
this process will occur for the rest of the days before being appended as timesteps
'''
dataset = get_numpy("IDIDIT.csv", "Lead Paragraph D1", 16) #start with first timestep of sentiment statistics
dataset = np.concatenate((dataset, get_numpy("IDIDIT.csv", "Prices D1", 6)), axis=1) #concatenate prices to sentiment
dataset = dataset.reshape(-1,22,1) #reshape to be 3D
dataset = dataset.swapaxes(1, 2) #swap axis for tensor to be in the correct LSTM format
#print(dataset.shape)
#print(get_numpy("IDIDIT.csv", "Prices D1", 6).shape)

#append rest of timesteps
for i in range(2,8):
    col = get_numpy("IDIDIT.csv", "Lead Paragraph D" + str(i), 16)
    col = np.concatenate((col, get_numpy("IDIDIT.csv", "Prices D" + str(i), 6)), axis=1)
    col = col.reshape(-1,22,1)
    col = col.swapaxes(1, 2)

    dataset = np.concatenate((dataset, col), axis=1)

print(dataset.shape)

#start preprocessing
norm = NDNormalizer() #class to normalize data
le = LabelEncoder()

comparison_df = pd.read_csv("IDIDIT.csv", usecols=[
        'comparison'], sep='\t') #read csv with usecols parameter

norm.fit(dataset)
normalized_dataset = norm.transform(dataset)

le.fit(comparison_df)
labels = le.transform(comparison_df)

#split dataset into testing and training (75/25)
split_data = int(len(normalized_dataset) * split_percent)
train_X = normalized_dataset[:split_data]
test_X = normalized_dataset[split_data:]

split_labels = int(len(labels) * split_percent)
train_Y = labels[:split_labels]
test_Y = labels[split_labels:]


#create model, start fitting and training it
#model = create_LSTM()


