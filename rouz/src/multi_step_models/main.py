"""
    @author Rouzbeh Yazdi
    Multi-step models. The model is trained and
    then predicts 24 hours ahead.
"""

import os
import pathlib
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from holidays.utils import country_holidays
from matplotlib.colors import TABLEAU_COLORS as colors
from matplotlib.ticker import NullFormatter, NullLocator
from tensorflow import keras

import helpers
import models
from window_generator import WindowGenerator

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Georgia",
        "font.size": 17,
        "lines.linewidth": 2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

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

 
def main():
    holidays = country_holidays("US", years=np.arange(2019, 2024, 1))


    train_df, test_df, val_df = helpers.read_data_Nic(holiday_calendar=holidays)
    train_df['time'] = pd.to_datetime(train_df['time'])
    train_df = train_df[train_df.time > datetime(2020,11, 3, 23, 0, 0)]

    ## feature scaling:
    train_time_ = train_df.pop("time")
    test_time_ = test_df.pop("time")
    val_time_ = val_df.pop("time")

    column_indices = {col: train_df.columns.get_loc(col) for col in train_df.columns}

    train_mean = train_df.mean(numeric_only=True, axis=0)
    train_std = train_df.std(numeric_only=True, axis=0)

    da_price_mean_train = train_df.DA_price.mean()
    da_price_std_train = train_df.DA_price.std()
    print(f"Mean and std of DA_price: ({da_price_mean_train},{da_price_std_train})")
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std


    """
        Only one windowing option here:
            Given 24 hours of inputs (input_width)
            predict 24 hours into the future
    """
    PATIENCE = 10
    MAX_EPOCHS = 200
    
    NFEATURES = train_df.shape[1]
    OUT_STEPS = 24
    CONVWIDTH = 5
    BATCHSIZE = 1024
    multi_window = WindowGenerator(input_width=24,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS,
                                   batch_size=BATCHSIZE,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df)
    print(multi_window)
    print(f"Number of features:{NFEATURES}")
    baseline_model = models.Baseline("Baseline")
    linear_model = models.LinearMultiStep("Linear", OUT_STEPS, NFEATURES)
    dense_model = models.DenseMultistep("Dense", OUT_STEPS, NFEATURES)
    cnn_model = models.ConvolutionalMultiStep("CNN", CONVWIDTH, OUT_STEPS, NFEATURES)


    ## Now train and test the models
    baseline_model.compile(loss=keras.losses.MeanSquaredError(),
                        metrics=[keras.metrics.RootMeanSquaredError()])

    print("*"*15,'\n',"Linear Model:")
    history_linear = models.compile_and_fit(linear_model, multi_window, 
                                            patience=PATIENCE, max_epochs=MAX_EPOCHS,
                                            learning_rate=0.0005)
    print("*"*15,'\n',"Dense Model:")
    history_dense = models.compile_and_fit(dense_model, multi_window, 
                                           patience=PATIENCE, max_epochs=MAX_EPOCHS,
                                           learning_rate=0.0005)
    print("*"*15,'\n',"CNN Model:")
    history_cnn = models.compile_and_fit(cnn_model, multi_window, 
                                         patience=PATIENCE, max_epochs=MAX_EPOCHS,
                                         learning_rate=0.0005)


    val_performance, performance = {}, {}
    #metric_index = linear_model.metrics_names.index('root_mean_squared_error')
    for model in [baseline_model, linear_model , dense_model, cnn_model]:
        name = model.name_
        print(f"Dealing with model: {name}")
        rms_err_val = get_rmse(model, multi_window.val, 
                          da_price_std_train, 
                          index=column_indices['DA_price'])
        print("Done with RMSE, validation set")
        rms_err_test = get_rmse(model, multi_window.test, 
                          da_price_std_train, 
                          index=column_indices['DA_price'])
        print("Done with RMSE, test set")
        val_performance[name] = rms_err_val 
        performance[name]     = rms_err_test

    fig1 = multi_window.plot(baseline_model, 
                             colors=['blue','green','orange'], title='Baseline')
    fig2 = multi_window.plot(linear_model, 
                             colors=['blue','green','orange'], title='Linear')
    fig3 = multi_window.plot(dense_model, 
                             colors=['blue','green','orange'], title='Dense')
    fig4 = multi_window.plot(cnn_model, 
                             colors=['blue','green','orange'], title='CNN')

    results = pd.DataFrame({'models':val_performance.keys(),
                            'Validation':val_performance.values(),
                            'Test':performance.values()})
    print(results)

    histories = {'Linear':history_linear,
                 'Dense':history_dense, 
                 'CNN': history_cnn}
    fig5, axes = plt.subplots(len(histories), 1)
    for hist, ax in zip(histories, axes):
        history = histories[hist]
        ax.plot([v * da_price_std_train**2 for v in history.history['loss']], 
                label='training', color='blue')
        ax.plot([v * da_price_std_train**2 for v in history.history['val_loss']], 
                label='validation', color='orange')
        ax.legend(loc='best')
        ax.text(0.4, 0.85, hist,transform=ax.transAxes)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xlabel('Epochs')
    fig5.supylabel('Loss Function (MSE)')

    width=0.3
    x = np.arange(len(performance))
    fig6, ax = plt.subplots(1, 1)
    ax.set_title(f"Multi-Step Models")
    ax.bar(x - 0.2, results['Validation'].values, width, label="Validation", color='orange')
    ax.bar(x + 0.2, results['Test'].values, width, label="Test", color='green')
    ax.set_xticks(ticks=x, labels=performance.keys(), rotation=90)
    ax.set_ylabel("RMSE (average over all outputs)")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    _ = ax.legend()

    plt.show()

if __name__=='__main__':
    tf.config.set_visible_devices([], "GPU")
    main()