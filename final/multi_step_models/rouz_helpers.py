from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from holidays.utils import country_holidays
from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def read_data(loc):
    holidays = country_holidays("US", years=np.arange(2019, 2024, 1))
    val_df = pd.read_csv(loc + "ordered_seasonal_validation_set.csv")
    test_df = pd.read_csv(loc + "ordered_test_set.csv")
    train_df = pd.read_csv(loc + "ordered_train_set.csv")
    selected_features = [
        'time', 'DA_price', 'RT_price', 'load', 
        'temp', 
        'dwpt',
        'nat_gas_spot_price', 
        'monthly_avg_NY_natgas_price', 
        'weekday', 'holiday', 'business_hour', 
        'avg_RT_price_prev_day', 'avg_actual_load_prev_day',

        'load(h-1)', 'load(h-2)', 
        'load(h-19)', 'load(h-20)', 
        'load(h-21)', 'load(h-22)', 
        'load(h-23)', #'load(h-24)', 
        'load(h-25)', 'load(h-26)',

        'price(h-1)', 'price(h-2)', 
        'price(h-19)', 'price(h-20)', 
        'price(h-21)', 'price(h-22)',
        'price(h-23)', 
        'price(h-25)', 
        'price(h-26)',

       'DA_price(t-1D)', 'DA_price(t-2D)', 
       'DA_price(t-3D)', 'DA_price(t-4D)', 
    #    'DA_price(t-5D)', 'DA_price(t-6D)',
    #    'DA_price(t-7D)',

    #    'RT_price(t-1D)', 'RT_price(t-2D)', 
    #   'RT_price(t-3D)', 'RT_price(t-4D)', 
    #    'RT_price(t-5D)', 'RT_price(t-6D)',
    #    'RT_price(t-7D)', 

       'load(t-1D)', 'load(t-2D)', 
       'load(t-3D)', 'load(t-4D)', 
       'load(t-5D)', 'load(t-6D)', 

       'hour_sin', 'hour_cos', 
       'day_sin', 'day_cos', 
       'day_of_week_sin', 'day_of_week_cos', 
       'month_sin', 'month_cos']

        
    for df in [val_df, test_df, train_df]:
        time_ = pd.to_datetime(df['time'])
        day = time_.dt.day
        df.loc[:,"hour_sin"] = sin_transformer(24).fit_transform(df["hour"])
        df.loc[:,"hour_cos"] = cos_transformer(24).fit_transform(df["hour"])
        df.loc[:,"day_sin"]  = sin_transformer(30.44).fit_transform(day)
        df.loc[:,"day_cos"]  = cos_transformer(30.44).fit_transform(day)
        df.loc[:,"day_of_week_sin"]  = sin_transformer(7).fit_transform(df["day_of_week"])
        df.loc[:,"day_of_week_cos"]  = cos_transformer(7).fit_transform(df["day_of_week"])
        df.loc[:,"month_sin"]  = sin_transformer(12).fit_transform(df["month"])
        df.loc[:,"month_cos"]  = cos_transformer(12).fit_transform(df["month"])
        df.loc[:,"holiday"] = [int(v in holidays) for v in df.date]

    return train_df[selected_features], val_df[selected_features], test_df[selected_features]


def process_data(train_df, val_df, test_df):
    train_df["time"] = pd.to_datetime(train_df["time"])
    train_df = train_df[train_df.time > datetime(2020, 11, 3, 23, 0, 0)]

    ## feature scaling:
    _ = train_df.pop("time")
    _ = test_df.pop("time")
    _ = val_df.pop("time")

    column_indices = {col: train_df.columns.get_loc(col) for col in train_df.columns}

    train_mean = train_df.mean(numeric_only=True, axis=0)
    train_std = train_df.std(numeric_only=True, axis=0)

    da_price_mean_train = train_df.DA_price.mean()
    da_price_std_train = train_df.DA_price.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return [
        train_df,
        val_df,
        test_df,
        da_price_mean_train,
        da_price_std_train,
        column_indices,
    ]


class WindowGenerator:
    """
    WindowGenerator class implemented in tensorflow
    tutorial on time series forecasting. Taken from
    (tensorflow website)[https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing].
    Used to generate tf.data.Dataset objects for training, cross-validation
    and test.
    """

    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        train_df: None,
        val_df: None,
        test_df: None,
        label_columns: List[str] = None,
        batch_size=32,
    ) -> None:
        ## store raw data:
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size_ = batch_size

        ## label column indices. label is the target for prediction/regression
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        ## time window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [
                    labels[:, :, self.column_indices[name]]
                    for name in self.label_columns
                ],
                axis=-1,
            )

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(
        self, model=None, colors=None, plot_col="DA_price", max_subplots=3, title=""
    ):
        inputs, labels = self.example
        # plt.figure()
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        fig, axes = plt.subplots(nrows=max_n, ncols=1, figsize=(12, 8))
        if title:
            axes[0].set_title(title)
        for n in range(max_n):
            ax = axes[n]
            # plt.subplot(max_n, 1, n+1)
            ax.set_ylabel("DA price [normed]")
            ax.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
                color=colors[0],
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            ax.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c=colors[1],
                s=64,
            )
            if model is not None:
                predictions = model(inputs)

            ax.scatter(
                self.label_indices,
                predictions[n, :, label_col_index],
                marker="X",
                edgecolors="k",
                label="Predictions",
                c=colors[2],
                s=64,
            )

            if n == 0:
                ax.legend()
            # ax.set_ylim(bottom=-2.5, top=2.5)
        return fig

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size_,
        )

        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result
        return result
