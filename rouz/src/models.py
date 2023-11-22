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
    def __init__(self, units=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(units)
    
    def call(self, inputs):
        return self.hidden(inputs)

class DenseModel(keras.Model):
    """
        Fully Connected, dense model. Applied to one day of input
        to one day of output. 
    """
    def __init__(self, units=24, **kwargs):
        super().__init__(**kwargs)
        self.layer_1_ = keras.layers.Dense(units, activation='relu', 
                            name=f'layer_1')
        self.output_ = keras.layers.Dense(1)

    def call(self, inputs):
        tmp_ = self.layer_1_(inputs)
        return self.output_(tmp_)

class DeepDenseModel(keras.Model):
    """
        Fully Connected, dense model. Applied to one day of input
        to one day of output. Has two dense layers.
    """
    def __init__(self, units=24, **kwargs):
        super().__init__(**kwargs)
        self.layer_1_ = keras.layers.Dense(units, activation='relu')
        self.layer_2_ = keras.layers.Dense(units, activation='relu')
        self.layer_3_ = keras.layers.Dense(units, activation='relu')
        self.output_ = keras.layers.Dense(1)

    def call(self, inputs):
        tmp_ = self.layer_1_(inputs)
        tmp_ = self.layer_2_(tmp_)
        tmp_ = self.layer_3_(tmp_)
        return self.output_(tmp_)

class MultiStepDenseModel(keras.Model):
    def __init__(self,  units=24, **kwargs):
        super().__init__(**kwargs)
        self.flattening_layer_ = keras.layers.Flatten()
        self.layer_0_ = keras.layers.Dense(units=units, activation='relu')
        self.layer_1_ = keras.layers.Dense(units=units, activation='relu')
        self.output_ = keras.layers.Dense(units=1)
        self.reshape_layer_ = keras.layers.Reshape([1,-1])

    def call(self, inputs):
        tmp_ = self.flattening_layer_(inputs)
        tmp_ = self.layer_0_(tmp_)
        tmp_ = self.layer_1_(tmp_)
        tmp_ = self.output_(tmp_)
        return self.reshape_layer_(tmp_)

class ConvolutionalModel(keras.Model):

    def __init__(self, convolution_window=12, units=12, filters=24, **kwargs):
        super().__init__(**kwargs)
        self.layer_0_ = keras.layers.Conv1D(filters=filters, 
                                            kernel_size=(convolution_window,),
                                            activation='relu')
        self.layer_1_ = keras.layers.Dense(units=units, activation='relu')
        self.layer_2_ = keras.layers.Dense(units=units, activation='relu')
        self.output_ = keras.layers.Dense(units=1)

    def call(self, inputs):
        tmp_ = self.layer_0_(inputs)
        tmp_ = self.layer_1_(tmp_)
        tmp_ = self.layer_2_(tmp_)
        return self.output_(tmp_)