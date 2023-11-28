"""
    @author: Rouzbeh Yazdi

    Helper functions for the neural network models
    used in the notebook. 
"""
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import FunctionTransformer
from holidays.utils import country_holidays

class WindowGenerator():
    """
        WindowGenerator class implemented in tensorflow 
        tutorial on time series forecasting. Taken from
        (tensorflow website)[https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing].
        Used to generate tf.data.Dataset objects for training, cross-validation 
        and test.
    """
    def __init__(self, input_width:int, label_width:int, shift:int,
                 train_df: None, val_df: None, test_df: None, 
                 label_columns: List[str], batch_size=32) -> None:

        ## store raw data:
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size_ = batch_size

        ## label column indices. label is the target for prediction/regression
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in 
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        ## time window parameters 
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size-self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self) -> str:
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
          labels = tf.stack(
              [labels[:, :, self.column_indices[name]] for name in self.label_columns],
              axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, colors=None, plot_col='DA_price', max_subplots=3, title=''):
        inputs, labels = self.example
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        fig, axes = plt.subplots(nrows=max_n, ncols=1, figsize=(9, 7))
        if title:
            axes[0].set_title(title)
        for n in range(max_n):
            ax = axes[n]
            ax.set_ylabel('DA price [normed]')
            ax.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10, color=colors[0])

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
        
            ax.scatter(self.label_indices, labels[n, :, label_col_index],
                      edgecolors='k', label='Labels', c=colors[1], s=64)
            if model is not None:
                predictions = model(inputs)

            ax.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c=colors[2], s=64)

            if n == 0:
                ax.legend()
        return fig

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size_,)
        
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
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.test))
            self._example = result
        return result
    
def print_correlations_to_files(df, loc="./", min_cutoff=0.1):

    for corr in ['pearson', 'kendall', 'spearman']:
        correlations = df.corrwith(df['DA_price'], method=corr, numeric_only=True, axis=0)
        largest_corrs = correlations[abs(correlations)>min_cutoff].sort_values(ascending=False)
        largest_corrs.name = f'{corr}_correlation'
        largest_corrs.index.name = 'features'
        fname = loc+f"{corr}_training_set.csv"
        largest_corrs.to_csv(fname, float_format="%0.5f")

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def read_data_Rouz(ratio_train, ratio_val, print_correlations=False):
    """
        read my own data, add new temporal features.
    """
    df = pd.read_csv("../processed_data/data_set_final_with_average_RT_price.csv")
    df['time'] = pd.to_datetime(df["time"])
    df['date'] = pd.to_datetime(df["date"]).dt.date
    df.drop(['RT_price'], inplace=True, axis=1)

    df.loc[:,"hour_sin"] = sin_transformer(24).fit_transform(df["hour"])
    df.loc[:,"hour_cos"] = cos_transformer(24).fit_transform(df["hour"])
    df.loc[:,"day_sin"]  = sin_transformer(30.44).fit_transform(df["day"])
    df.loc[:,"day_cos"]  = cos_transformer(20.44).fit_transform(df["day"])
    df.loc[:,"day_of_week_sin"]  = sin_transformer(7).fit_transform(df["day_of_week"])
    df.loc[:,"day_of_week_cos"]  = cos_transformer(7).fit_transform(df["day_of_week"])
    df.loc[:,"month_sin"]  = sin_transformer(12).fit_transform(df["month"])
    df.loc[:,"month_cos"]  = cos_transformer(12).fit_transform(df["month"])

    features_to_drop = ['hour', 'day_of_week', 'day', 'season',
                        'month']

    data = df.drop(features_to_drop,axis=1)
    n = len(data)
    indx_trn = int(n*ratio_train)
    indx_val = indx_trn + int(n*ratio_val)
    train_df = data[0 : indx_trn-1]
    val_df   = data[indx_trn : indx_val-1]
    test_df  = data[indx_val : ]
    if print_correlations:
        print_correlations_to_files(train_df, loc='./', min_cutoff=0.1)

    return train_df, val_df, test_df

def read_data_Nic():
    holiday_calendar = country_holidays("US", years=np.arange(2019, 2024, 1))

    val_df = pd.read_csv("../final_datasets/smaller_ordered_seasonal_validation_set.csv")
    test_df = pd.read_csv("../final_datasets/smaller_ordered_test_set.csv")
    train_df = pd.read_csv("../final_datasets/larger_ordered_train_set.csv")

    for df in [val_df, test_df, train_df]:
        time_ = pd.to_datetime(df['time'])
        day = time_.dt.day
        df.loc[:,"hour_sin"] = sin_transformer(24).fit_transform(df["hour"])
        df.loc[:,"hour_cos"] = cos_transformer(24).fit_transform(df["hour"])
        df.loc[:,"day_sin"]  = sin_transformer(30.44).fit_transform(day)
        df.loc[:,"day_cos"]  = cos_transformer(20.44).fit_transform(day)
        df.loc[:,"day_of_week_sin"]  = sin_transformer(7).fit_transform(df["day_of_week"])
        df.loc[:,"day_of_week_cos"]  = cos_transformer(7).fit_transform(df["day_of_week"])
        df.loc[:,"month_sin"]  = sin_transformer(12).fit_transform(df["month"])
        df.loc[:,"month_cos"]  = cos_transformer(12).fit_transform(df["month"])

        df.loc[:,"holiday"] = [int(v in holiday_calendar) for v in df.date]

        features_to_drop = ['minute','hour', 'day_of_week', 'season', 
                            'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0',
                            'month', 'date']

        df.drop(features_to_drop, axis=1, inplace=True)

    return train_df, val_df, test_df

