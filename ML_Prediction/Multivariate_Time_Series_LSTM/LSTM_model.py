#python
from gettext import npgettext #might be superfluous
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
from keras.layers import LSTM


#Preproccess data
def get_numpy(csv_name, label):
    #read csv with usecols parameter
    #not sure whether pop should instead be used
    df = pd.read_csv(csv_name, usecols=[
        label], sep='\t')

    #this is to make an array with shape (1,16) so that concatenation can occyr
    arr = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],dtype=float)

    for i in range(len(df)):
        #take the object out of the cell and take as item
        ind = str(df.loc[i].item())
        #remove the brackets as np.fromstring does not work with brackets
        ind = ind.replace('[','')
        ind = ind.replace(']','')
        ind = ind.replace(', ', '|')
        #concatenate
        arr = np.concatenate((arr, [np.fromstring(ind, dtype=float, sep='|')]), axis=0)
    arr = np.delete(arr, 0, 0)
    return arr




print(get_numpy('IDIDIT.csv',  'Snippet D1'))