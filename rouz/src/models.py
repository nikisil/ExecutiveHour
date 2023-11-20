"""
   @author: Rouzbeh Yazdi

   Goal is to do a time-series forecast of the day-ahead price of 
   electricity for the next 24 hours.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def compile_and_fit(model, window, patience=2, max_epochs=20, learning_rate=1e-2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=max_epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


class Baseline(tf.keras.Model):
    """
        The baseline model is one of _no change_. 
        It predicts the day ahead price of tomorrow (evaluated today)
        to be the day ahead price of yesterday. 
    """
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

class Linear(keras.Model):
    """
        Simple linear model. Given one day of input,
        predict the next day. 
    """
    def __init__(self, units=24, **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(units)
    
    def call(self, inputs):
        return self.hidden(inputs)

class FullyConnectedDense(keras.Model):
    """
        Fully Connected, dense model. Applied to one day of input
        to one day of output. 
    """
    def __init__(self, units=24, layers=2, **kwargs):
        super().__init__(**kwargs)
        self.num_layers_ = layers 
        self.layers_ = []
        if layers > 1:
            for layer in range(0, layers):
                self.layers_.append(keras.layers.Dense(units, 
                                                       activation='softsign', 
                                                       name=f'layer_{layer}'))
        self.output_ = keras.layers.Dense(units)

    def call(self, inputs):
        if self.num_layers_ > 1:
            intermediates = [self.layers_[0](inputs)]
            for layer in range(1, self.num_layers_):
                intermediates.append(self.layers_[layer](intermediates[0]))
            return self.output_(intermediates[-1])
        else:
            return self.output_(inputs)

class FullyConnectedDenseMultiStep(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return "false"