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
        'Snippet D1','Lead Paragraph D1','Prices D1',
        'Snippet D2','Lead Paragraph D2','Prices D2',
        'Snippet D3','Lead Paragraph D3','Prices D3',
        'Snippet D4','Lead Paragraph D4','Prices D4',
        'Snippet D5','Lead Paragraph D5','Prices D5',
        'Snippet D6','Lead Paragraph D6','Prices D6',
        'Snippet D7','Lead Paragraph D7','Prices D7'
        ], sep='\t').to_numpy()
    print(df.shape)

preprocess('IDIDIT.csv')