"""
   @author: Rouzbeh Yazdi

   Goal is to do a time-series forecast of the day-ahead price of 
   electricity for the next 24 hours.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def compile_and_fit(model, window, patience=2, 
                    max_epochs=20, learning_rate=1e-2,
                    fname='./tmp.keras'
                    ):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    save_best = keras.callbacks.ModelCheckpoint(fname, 
                                                save_best_only=True, 
                                                monitor='val_loss', 
                                                mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping, save_best])
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
    def __init__(self, num_layers = 1, units=24, **kwargs):
        super().__init__(**kwargs)
        self.n_layers_ = num_layers
        self.layers_ = keras.Sequential()
        if num_layers == 1:
            self.layers_.add(keras.layers.Dense(units, activation='relu'))
        else:
            for layer in range(num_layers):
                self.layers_.add(keras.layers.Dense(units=units[layer], activation='relu'))
            self.layers_.add(keras.layers.Dense(1))

    def call(self, inputs):
        tmp_ = self.layers_(inputs)
        return tmp_

class ConvolutionalModel(keras.Model):

    def __init__(self, c_window=12, add_dense=True,
                 units=24, filters=24, **kwargs):
        super().__init__(**kwargs)

        self.layer_0_ = keras.layers.Conv1D(filters=filters, 
                                            kernel_size=(c_window,),
                                            activation='relu')
        self.add_dense = add_dense
        if add_dense:
            self.layer_2_ = keras.layers.Dense(units=units, activation='relu')
        self.output_ = keras.layers.Dense(units=1)

    def call(self, inputs):
        tmp_ = self.layer_0_(inputs)
        if self.add_dense:
            tmp_ = self.layer_2_(tmp_)
        return self.output_(tmp_)

class RecurrentModel(keras.Model):

    def __init__(self, layer='LSTM', units=32, activation='tanh', 
                 add_dense=False, **kwargs):
        super().__init__(**kwargs)
        self.include_dense_ = add_dense
        self.layers_ = keras.Sequential()
        if add_dense:
            self.layers_.add(keras.layers.Dense(units=20, activation='relu'))
        if layer == 'LSTM':
            self.layers_.add(keras.layers.LSTM(units=units, activation=activation, return_sequences=True))
        else:
            self.layers_.add(keras.layers.GRU(units=units,activation=activation, return_sequences=True))
        self.layers_.add(keras.layers.Dense(1))

    def call(self, inputs):
        return self.layers_(inputs)

class ResidualNetwork(keras.Model):
    
    def __init__(self, nfeatures=41):
        super().__init__()
        self.model = keras.Sequential([
                        keras.layers.Dense(units=256, activation='relu'),
                        keras.layers.Dense(units=128, activation='relu'), 
                        keras.layers.LSTM(64, return_sequences=True),
                        keras.layers.Dense(
                            nfeatures,
                            # The predicted deltas should start small.
                            # Therefore, initialize the output layer with zeros.
                            kernel_initializer=tf.initializers.zeros())
                    ])
    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each time step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta