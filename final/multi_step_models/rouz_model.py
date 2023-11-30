import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def get_rmse(model, dataset, std, index=0):
    rms_err, num = 0, 0
    
    for x, y in dataset:

        ## make prediction 
        y_pred = model.predict_on_batch(x)

        ## get the correct column
        y_pred = y_pred[:, :, index]
        y_true = y[:, :, index]

        batch_err = keras.metrics.mean_absolute_error(y_true, y_pred)
        rms_err += sum(batch_err)
        num += batch_err.shape[0] ## size of the batch: number of values.
    ## scale by the std of the quantity so the units are correct
    rms_err = std * np.sqrt(rms_err/num)
    return rms_err


def compile_and_fit(
    model,
    window,
    patience=2,
    max_epochs=20,
    learning_rate=5e-4,
    save=False,
    fname="./tmp.keras",
):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )
    callbacks = [early_stopping]
    if save:
        save_best = keras.callbacks.ModelCheckpoint(
            fname, save_best_only=True, monitor="val_loss", mode="min"
        )
        callbacks.append(save_best)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    history = model.fit(
        window.train,
        epochs=max_epochs,
        validation_data=window.val,
        callbacks=callbacks,
    )
    return history


class Baseline(keras.Model):
    def __init__(self, name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name_ = name

    def call(self, inputs):
        return inputs


class LinearMultiStep(keras.Model):
    def __init__(self, name, output_width, nfeatures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_ = name
        self.layers_ = keras.Sequential()  ## Sequentially apply-->
        ## get the last time step:
        self.layers_.add(keras.layers.Lambda(lambda x: x[:, -1:, :]))
        ## apply a dense layer with this (output_width * nfeature)
        ## many neurons
        self.layers_.add(
            keras.layers.Dense(
                output_width * nfeatures, kernel_initializer=tf.initializers.zeros()
            ),
        )
        ## reshape appropriately
        self.layers_.add(keras.layers.Reshape([output_width, nfeatures]))

    def call(self, inputs):
        return self.layers_(inputs)


class DenseMultistep(keras.Model):
    def __init__(self, name, output_width, nfeatures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_ = name
        self.layers_ = keras.Sequential()  ## Sequentially apply-->
        ## get the last time step:
        self.layers_.add(keras.layers.Lambda(lambda x: x[:, -1:, :]))
        ## A dense layer with ReLU:
        self.layers_.add(keras.layers.Dense(256, activation="relu"))
        ## apply a dense layer with this (output_width * nfeature)
        ## many neurons
        self.layers_.add(
            keras.layers.Dense(
                output_width * nfeatures, kernel_initializer=tf.initializers.zeros()
            ),
        )
        ## reshape appropriately
        self.layers_.add(keras.layers.Reshape([output_width, nfeatures]))

    def call(self, inputs):
        return self.layers_(inputs)


class ConvolutionalMultiStep(keras.Model):
    def __init__(self, name, conv_width, output_width, nfeatures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_ = name
        self.layers_ = keras.Sequential()
        ## take the previous conv_width time steps:
        self.layers_.add(keras.layers.Lambda(lambda x: x[:, -conv_width:, :]))
        ## apply a convolution layer
        self.layers_.add(
            keras.layers.Conv1D(256, activation="relu", kernel_size=(conv_width))
        )
        ## apply a linear dense layer for prediction
        self.layers_.add(
            keras.layers.Dense(
                output_width * nfeatures, kernel_initializer=tf.initializers.zeros()
            )
        )
        ## reshape to the right format
        self.layers_.add(keras.layers.Reshape([output_width, nfeatures]))

    def call(self, inputs):
        return self.layers_(inputs)
