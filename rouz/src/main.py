"""
    @author Rouzbeh Yazdi
    Single step models. The model is trained and
    then predicts one hour ahead.
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


def get_rmse(y_pred, y_true, is_residual=False, index=0):
    y_true = tf.reshape(y_true, [-1]).numpy()
    if not is_residual:
        y_pred = tf.reshape(y_pred, [-1]).numpy()
    else:
        y_pred = y_pred[:, :, index]
        y_pred = tf.reshape(y_pred, [-1])

    err = sum((y_true - y_pred) ** 2)
    num = len(y_true)
    return err, num


def main(holidays=None, multistep=False):
    input_color, label_color, prediction_color = (
        colors["tab:blue"],
        colors["tab:orange"],
        colors["tab:green"],
    )
    train_color, validation_color, test_color = (
        colors["tab:blue"],
        colors["tab:green"],
        colors["tab:red"],
    )

    # train_df, test_df, val_df = helpers.read_data_Nic(holiday_calendar=holidays)
    # train_df['time'] = pd.to_datetime(train_df['time'])
    # train_df = train_df[train_df.time > datetime(2020,11, 3, 23, 0, 0)]

    train_df, test_df, val_df = helpers.read_data_Rouz(0.7, 0.2)

    ## feature scaling:
    train_time_ = train_df.pop("time")
    test_time_ = test_df.pop("time")
    val_time_ = val_df.pop("time")

    for df in [train_df, test_df, val_df]:
        df.drop(["date"], inplace=True, axis=1)

    column_indices = {col: train_df.columns.get_loc(col) for col in train_df.columns}

    train_mean = train_df.mean(numeric_only=True, axis=0)
    train_std = train_df.std(numeric_only=True, axis=0)

    da_price_mean_train = train_df.DA_price.mean()
    da_price_std_train = train_df.DA_price.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    """
        Window work:
            create windows for the data to be fed to the models
            this uses the WindowGenerator class defined in 
            window_generator.py, copied from the TensorFlow 
            webpage.
    """

    SHIFT = 24 if multistep else 1
    output_type = "multi" if SHIFT > 1 else "single"
    print(f"shift: {SHIFT}, model output type: {output_type}")
    BATCH_SIZE = 128
    ## window no. 1: a narrow window of time, one hour of data, one hour prediction
    narrow_window = WindowGenerator(
        input_width=1,
        label_width=1,
        shift=1,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=["DA_price"],
        batch_size=BATCH_SIZE,
    )
    ## window no. 2: wide window, passes 24 hours of data for prediction
    wide_window = WindowGenerator(
        input_width=24,
        label_width=24,
        shift=SHIFT,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=["DA_price"],
        batch_size=BATCH_SIZE,
    )

    ## window no.3: windowing for the convolutional model
    CONV_WIDTH = 6
    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    conv_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=SHIFT,
        label_columns=["DA_price"],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    """
        Models:
            1. Baseline model where DA_price(h) = DA_price(h-1)
            2. Linear model: single neuron
            3. Dense model: one dense layer with activation + 1 linear neuron
            4. Deep, Dense model: multiple dense layers + 1 linear neuron
    """
    MAX_EPOCHS, PATIENCE = 100, 8
    model_names = {
        "Baseline": models.Baseline(label_index=column_indices["DA_price"]),
        "Linear": models.Linear(),
    }
    for u in range(1, 61, 10):
        model_names[f"Dense (units={u})"] = models.DenseModel(units=u)

    deep_dense_units = {2: [40, 20], 3: [40, 20, 10], 4: [40, 20, 10, 5]}
    for ilayer in range(2, 5, 1):
        model_names[f"Deep Dense {ilayer} layers"] = models.DeepDenseModel(
            num_layers=ilayer, units=deep_dense_units[ilayer]
        )

    model_names["Conv no dense"] = models.ConvolutionalModel(
        c_window=CONV_WIDTH, add_dense=False, filters=6
    )

    model_names["Conv with dense (units=6)"] = models.ConvolutionalModel(
        c_window=CONV_WIDTH, add_dense=True, units=6, filters=6
    )

    model_names["LSTM"] = models.RecurrentModel(layer="LSTM", units=32)
    model_names["GRU"] = models.RecurrentModel(layer="GRU", units=32)
    model_names["Dense and LSTM"] = models.RecurrentModel(
        layer="LSTM", units=8, add_dense=True
    )
    model_names["Residual LSTM"] = models.ResidualNetwork(nfeatures=len(column_indices))
    histories, val_perf, perf = {}, {}, {}

    plt_save_loc = pathlib.Path(f"../plots/dat_set_rouz/{output_type}_step_plots/")
    plt_save_loc.mkdir(parents=True, exist_ok=True)

    wide_validation_ds = wide_window.val
    wide_test_ds = wide_window.test

    conv_validation_ds = conv_window.val
    conv_test_ds = conv_window.test

    num, root_mean_sq_err_val, root_mean_sq_err_test = 0, 0, 0
    for model_name, model in model_names.items():
        print(f"*** MODEL NAME: {model_name} ***")
        if model_name == "Baseline":
            model.compile(
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.MeanAbsoluteError()],
            )

            root_mean_sq_err_val = 0
            num = 0
            for x, y in wide_validation_ds:
                y_tru = y * da_price_std_train + da_price_mean_train
                y_pred = model.predict(x, verbose=0)
                y_pred = y_pred * da_price_std_train + da_price_mean_train
                ierr, inum = get_rmse(y_pred, y_tru)
                root_mean_sq_err_val += ierr
                num += inum

            val_perf[model_name] = np.sqrt(root_mean_sq_err_val / num)

            root_mean_sq_err_test = 0
            num = 0
            for x, y in wide_test_ds:
                y_tru = y * da_price_std_train + da_price_mean_train
                y_pred = model.predict(x, verbose=0)
                y_pred = y_pred * da_price_std_train + da_price_mean_train
                ierr, inum = get_rmse(y_pred, y_tru)
                root_mean_sq_err_test += ierr
                num += inum

            perf[model_name] = np.sqrt(root_mean_sq_err_test / num)

            fig = wide_window.plot(
                model,
                max_subplots=3,
                title="Model: Baseline",
                colors=[input_color, label_color, prediction_color],
            )
            fname = plt_save_loc / "baseline_model_examples.pdf"
            fig.savefig(fname, dpi=200)
            plt.close()
            continue

        is_conv_model = True if "Conv" in model_name else False
        is_recurrent = True if "LSTM" in model_name or "GRU" in model_name else False
        is_residual = True if "Residual" in model_name else False

        window = conv_window if is_conv_model else narrow_window
        window = window if not is_recurrent else wide_window
        window = wide_window if is_residual else window

        validation_dataset = conv_validation_ds if is_conv_model else wide_validation_ds
        test_dataset = conv_test_ds if is_conv_model else wide_validation_ds
        tmp = model_name.replace(" ", "_")
        model_save_loc = pathlib.Path(
            f"../models/dat_set_rouz/{output_type}_step/{tmp}/"
        )
        model_save_loc.mkdir(parents=True, exist_ok=True)
        model_fname = model_save_loc / f"{tmp}.keras"
        model_fname = os.path.dirname(model_fname.resolve())
        histories[model_name] = models.compile_and_fit(
            model,
            window=window,
            patience=PATIENCE,
            max_epochs=MAX_EPOCHS,
            fname=model_fname,
        )

        root_mean_sq_err_val = 0
        num = 0
        for x, y in validation_dataset:
            y_tru = y * da_price_std_train + da_price_mean_train
            y_pred = model.predict(x, verbose=0)
            y_pred = y_pred * da_price_std_train + da_price_mean_train
            ierr, inum = get_rmse(
                y_pred, y_tru, is_residual, index=column_indices["DA_price"]
            )
            root_mean_sq_err_val += ierr
            num += inum

        val_perf[model_name] = np.sqrt(root_mean_sq_err_val / num)

        root_mean_sq_err_test = 0
        num = 0
        for x, y in test_dataset:
            y_tru = y * da_price_std_train + da_price_mean_train
            y_pred = model.predict(x, verbose=0)
            y_pred = y_pred * da_price_std_train + da_price_mean_train
            ierr, inum = get_rmse(
                y_pred, y_tru, is_residual, index=column_indices["DA_price"]
            )
            root_mean_sq_err_test += ierr
            num += inum

        perf[model_name] = np.sqrt(root_mean_sq_err_test / num)

        window = conv_window if is_conv_model else wide_window
        fig = window.plot(
            model,
            max_subplots=3,
            title=f"{model_name}",
            colors=[input_color, label_color, prediction_color],
        )

        fname = plt_save_loc / f"{tmp}_model_examples.pdf"
        fig.savefig(fname, dpi=200)
        plt.close()

        if model_name == "Linear":
            ## use the linear model's simplicity to get a sense of the
            ## weight assigned to each feature:
            fig, ax = plt.subplots(
                1,
                1,
                figsize=(16, 9),
                gridspec_kw={"left": 0.05, "right": 0.98, "top": 0.95, "bottom": 0.35},
            )
            features = [
                [x, y]
                for x, y in zip(
                    train_df.columns.values, model.layers[0].kernel[:, 0].numpy()
                )
            ]
            features_sorted = sorted(features, key=lambda x: x[1], reverse=True)
            x = [v[0] for v in features_sorted]
            y = [v[1] for v in features_sorted]
            ax.bar(
                x=range(len(train_df.columns)), height=y
            )  # model.layers[0].kernel[:,0].numpy())
            ax.set_xticks(range(len(train_df.columns)))
            _ = ax.set_xticklabels(x, rotation=90)  # train_df.columns, rotation=90)
            ax.set_title("Linear model's fitted weights")
            ax.set_ylabel("Weight")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())

            fname = plt_save_loc / "linear_weights.pdf"
            fig.savefig(fname.resolve(), dpi=200)
            plt.close()

    print("**** Validation ****")
    for key, val in val_perf.items():
        print(key, val)
    print("**** Test ****")
    for key, val in perf.items():
        print(key, val)

    hist_save_loc = pathlib.Path(f"../histories/dat_set_rouz/{output_type}_step_plots/")
    hist_save_loc.mkdir(parents=True, exist_ok=True)

    for model, hist in histories.items():
        fname = hist_save_loc / f"{model}.csv"
        pd.DataFrame.from_dict(hist.history).to_csv(fname.resolve(), index=False)

    """
        Plot the performances of the models:
    """
    ## Plot the performance of the various models:
    x = np.arange(len(perf))
    width = 0.3
    val_mae = val_perf.values()
    test_mae = perf.values()

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{output_type}-step models")
    ax.bar(x - 0.17, val_mae, width, label="Validation", color=validation_color)
    ax.bar(x + 0.17, test_mae, width, label="Test", color=test_color)
    ax.set_xticks(ticks=x, labels=perf.keys(), rotation=90)
    ax.set_ylabel("RMSE (average over all outputs)")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    _ = ax.legend()

    # # print(history_linear.params, history_linear.history.keys())
    # fig2, axes2 = plt.subplots(len(histories), 1, sharex=True)
    # for ax, model_name in zip(axes2, histories):
    #     history = histories[model_name]
    #     ax.plot(history.history['loss'], label='training', color=train_color)
    #     ax.plot(history.history['val_loss'], label='validation', color=validation_color)
    #     ax.legend(loc='best')
    #     ax.text(0.4, 0.85, model_name,transform=ax.transAxes)
    #     ax.xaxis.set_minor_locator(NullLocator())
    #     ax.xaxis.set_minor_formatter(NullFormatter())
    # axes2[-1].set_xlabel('Epochs')
    # fig2.supylabel('Loss Function (MSE)')
    plt.show()


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    holidays = country_holidays("US", years=np.arange(2019, 2024, 1))
    multistep = int(sys.argv[1])
    main(holidays, multistep=multistep)
