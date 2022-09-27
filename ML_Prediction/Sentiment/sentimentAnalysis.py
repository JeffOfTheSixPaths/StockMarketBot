import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow.keras
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import os



def make_sentiment_model(train_dataset):
    encoder = tf.keras.layers.TextVectorization()
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            mask_zero=True),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    return model

#as of writing, this has 96.875% accuracy for extremely polar comments with our weights

