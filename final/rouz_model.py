import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

class Baseline(keras.Model):
    """
        The baseline model is one of _no change_.
        It predicts the day ahead price of the next 
        time step to be the day ahead price of the 
        current time step.
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
        Simple linear model. 
    """

    def __init__(self, units=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(units)

    def call(self, inputs):
        return self.hidden(inputs)

class DeepDenseModel(keras.Model):
    """
        Fully Connected, dense model. Applied to one day of input
    to one day of output. Has two dense layers.
    """

    def __init__(self, num_layers=1, units=24, **kwargs):
        super().__init__(**kwargs)
        self.n_layers_ = num_layers
        self.layers_ = keras.Sequential()
        if num_layers == 1:
            self.layers_.add(keras.layers.Dense(units, activation="relu"))
        else:
            for layer in range(num_layers):
                self.layers_.add(
                    keras.layers.Dense(units=units[layer], activation="relu")
                )
            self.layers_.add(keras.layers.Dense(1))

    def call(self, inputs):
        tmp_ = self.layers_(inputs)
        return tmp_

class ResidualNetwork(keras.Model):
    def __init__(self, nfeatures=41):
        super().__init__()
        self.model = keras.Sequential(
            [
                keras.layers.GRU(256, return_sequences=True),
                keras.layers.Dense(
                    nfeatures,
                    kernel_initializer=tf.initializers.zeros(),
                ),
            ]
        )

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        return inputs + delta


def compile_and_fit(model, window, patience=2, 
                    max_epochs=20, learning_rate=1e-2, 
                    save_model=False,
                    fname="./tmp.keras"):
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )
    cbacks = [early_stopping]
    if save_model:
        save_best = keras.callbacks.ModelCheckpoint(
            fname, save_best_only=True, monitor="val_loss", mode="min"
        )
        cbacks.append(save_best)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    history = model.fit(
        window.train,
        epochs=max_epochs,
        validation_data=window.val,
        callbacks=cbacks,
        verbose=0
    )
    return history


def get_rmse(model, dataset, mean, std, is_residual=False, index=0):
    rms_err, num = 0, 0

    for x, y in dataset:
        y_tru = y * std + mean 
        y_pred = model.predict(x, verbose=0)
        y_pred = y_pred * std + mean
        y_tru = tf.reshape(y_tru, [-1]).numpy()
        if not is_residual:
            y_pred = tf.reshape(y_pred, [-1]).numpy()
        else:
            y_pred = y_pred[:, :, index]
            y_pred = tf.reshape(y_pred, [-1])
        rms_err += sum((y_tru - y_pred) ** 2)
        num += len(y_tru)

    return np.sqrt(rms_err/num)

 