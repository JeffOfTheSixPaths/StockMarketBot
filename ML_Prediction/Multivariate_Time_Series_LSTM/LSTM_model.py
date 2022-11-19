#python
from gettext import npgettext #might be superfluous
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import datasets, layers, models
import pandas as pd
import numpy as np
from keras.layers import LSTM


#Preproccess data
def preprocess(csv_name):
    df = pd.read_csv(csv_name, usecols=[
        'Snippet D1'], sep='\t').to_numpy()
    
    
    print(df)

preprocess('IDIDIT.csv')