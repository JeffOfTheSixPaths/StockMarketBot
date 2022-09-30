from base64 import standard_b64decode
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow.keras
import pandas as pd
import numpy as np
import json
import yfinance as yf
import os
import sys
import_path= '../ML_Prediction/Sentiment'
sys.path.insert(1, import_path)

import sentimentAnalysis
import Statistics

weights_path = '../sentiment_model_weights/cp.cpkt'
weights_dir = os.path.dirname(weights_path)
latest = tf.train.latest_checkpoint(weights_dir)

train_text = 'sentiment_model_weights/sentiment_text.csv'
train_sentiment = 'sentiment_model_weights/sentiment_sentiment.npy'
s140_text = 'sentiment_model_weights/sentiment140_text.csv'
s140_sentiment = 'sentiment_model_weights/sentiment140_sentiment.npy'

# <something>_text holds the sentences and <something>_sentiment has the sentiment to <sometime>_text's sentences
print("loading the data")
train_text = pd.read_csv(train_text)
train_sentiment = np.load(train_sentiment)
train_text.pop('Unnamed: 0')
s140_text = pd.read_csv(s140_text)
s140_sentiment = np.load(s140_sentiment)
s140_text.pop('Unnamed: 0')
print("loaded the data")

print("converting data to a usable form for the ai")
train_text = tf.convert_to_tensor(train_text)
s140_tensor = tf.convert_to_tensor(s140_text)
train_text = tf.concat([train_text,s140_tensor], 0)
train_sentiment = tf.convert_to_tensor(train_sentiment)
train_dataset = tf.data.Dataset.from_tensor_slices((train_text,train_sentiment))
print("converted")

print('creating the model and encoding the data')
model = sentimentAnalysis.make_sentiment_model(train_dataset)
print('loading weights')
model.load_weights(latest)
print('loaded weights')
def predict(sentence: str):
    return model.predict(np.array([sentence]))

test_sentences = [
                    'I love this',
                    'I hate this',
                    'Gamers are oppressed',
                    'I fucking love dogs', #using the f-word here as it can be used with different connotations in different sentences
                    'I fucking hate dogs', #it can also be used as a stronger form of "really"
                    'I really love dogs',
                    'I really hate dogs',
                    'I love dogs',
                    'I hate dogs',
                    'Rubiks cubes are a neat little thing',
                    'I really like eating meat'
                ]

print('Printing test sentences \n')
for sentence in test_sentences: #just to test if the model loaded correctly, it should predict on the interval [0,1], but it frequently leaves that interval
    print(sentence)
    print(predict(sentence))
    print('\n')
