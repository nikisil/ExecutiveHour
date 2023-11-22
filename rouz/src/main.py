from window_generator import WindowGenerator
import models
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import numpy as np
from matplotlib.colors import TABLEAU_COLORS as colors

input_color, label_color, prediction_color = colors['tab:blue'], colors['tab:orange'], colors['tab:green'] 
train_color, validation_color, test_color = colors['tab:blue'], colors['tab:green'], colors['tab:red']


raw_data = pd.read_csv("../processed_data/data_set_final_with_average_RT_price.csv")
raw_data['time'] = pd.to_datetime(raw_data["time"])
raw_data['date'] = pd.to_datetime(raw_data["date"]).dt.date
raw_data.drop(['RT_price'], inplace=True, axis=1)


features = ['time','DA_price','load','temp','dwpt',
            'nat_gas_spot_price','monthly_avg_NY_natgas_price',
            'price(h-1)','price(h-2)','price(h-19)','price(h-20)',
            'price(h-21)', 'price(h-22)', 'price(h-23)',
            'price(h-24)','price(h-25)','price(h-49)','price(h-168)',
            'load(h-1)' , 'load(h-2)' ,'load(h-19)','load(h-20)',
            'load(h-21)', 'load(h-22)', 'load(h-23)',
            'load(h-24)','load(h-25)','load(h-49)','load(h-168)',
            'hour','weekday','month','day_of_week','holiday',
            'business_hour','season',
            'avg_RT_price_prev_day','avg_actual_load_prev_day']

data = raw_data[features]

time_ = data.pop('time')
## now split the data set to train, validation, test according
## to 70, 20, 10 percent
n = len(data)

## ratio_test would be 1 - ratio_train - ratio_val
ratio_train, ratio_val = 0.7, 0.9

train_df = data[0:int(n*ratio_train)]
val_df   = data[int(n*ratio_train):int(n*ratio_val)]
test_df  = data[int(n*ratio_val):]

## feature scaling:
train_mean = train_df.mean(numeric_only=True, axis=0)
train_std  = train_df.std(numeric_only=True, axis=0)

train_df = (train_df - train_mean) / train_std
val_df   = (val_df   - train_mean) / train_std
test_df  = (test_df  - train_mean) / train_std

column_indices = {name: i for i, name in enumerate(data.columns)}

"""
    Window work:
        create windows for the data to be fed to the models
        this uses the WindowGenerator class defined in 
        window_generator.py, copied from the TensorFlow 
        webpage.
"""
## window no. 1: a narrow window of time, one hour of data, one hour prediction
BATCH_SIZE=256
narrow_window = WindowGenerator(input_width=1, 
                             label_width=1, 
                             shift=1, 
                             train_df=train_df, val_df=val_df,
                             test_df=test_df, label_columns=['DA_price'],
                             batch_size=BATCH_SIZE)
## window no. 2: wide window, passes 24 hours of data for prediction 
wide_window = WindowGenerator(input_width=24, 
                             label_width=24, 
                             shift=1, 
                             train_df=train_df, val_df=val_df,
                             test_df=test_df, label_columns=['DA_price'],
                             batch_size=BATCH_SIZE)
## window no.3: use multiple time steps in the training, 
MULTISTEP_WIDTH = 6
multistep_window = WindowGenerator(input_width=MULTISTEP_WIDTH, 
                             label_width=1, 
                             shift=1, 
                             train_df=train_df, val_df=val_df,
                             test_df=test_df, label_columns=['DA_price'],
                             batch_size=BATCH_SIZE)
## window no.4: windowing for the convolutional model
CONV_WIDTH  = 24
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
conv_window = WindowGenerator(input_width=INPUT_WIDTH,
     label_width=LABEL_WIDTH,
     shift=1,
     label_columns=['DA_price'],
     train_df=train_df, val_df=val_df, test_df=test_df)


"""
    Models:
        1. Baseline model where DA_price(h) = DA_price(h-1)
        2. Linear model: single neuron
        3. Dense model: one dense layer with activation + 1 linear neuron
        4. Deep, Dense model: multiple dense layers + 1 linear neuron
"""
MAX_EPOCHS, PATIENCE = 100, 10
## 1. Baseline
baseline = models.Baseline(label_index=column_indices['DA_price'])
baseline.compile(loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

## 2. Linear:
linear = models.Linear()
hist_linear = models.compile_and_fit(linear, window=narrow_window, 
                                        patience=PATIENCE, 
                                        max_epochs=MAX_EPOCHS)


## 3. simple deep model
deep_simple = models.DenseModel()
hist_deep_simple = models.compile_and_fit(deep_simple, window=narrow_window,
                                          patience=PATIENCE, 
                                          max_epochs=MAX_EPOCHS)

## 4. more complex deep model
deep_complex = models.DeepDenseModel()
hist_deep_complex = models.compile_and_fit(deep_complex, window=narrow_window,
                                           patience=PATIENCE,
                                           max_epochs=MAX_EPOCHS)

## 5. Multi-step dense model
deep_multistep = models.MultiStepDenseModel()
hist_deep_multistep = models.compile_and_fit(deep_multistep, window=multistep_window,
                                           patience=PATIENCE,
                                           max_epochs=MAX_EPOCHS)

## 6. Convolutional model
deep_conv = models.ConvolutionalModel(convolution_window=CONV_WIDTH,
                                      units=12,
                                      filters=24)
hist_deep_conv = models.compile_and_fit(deep_conv, window=conv_window,
                                        patience=PATIENCE,
                                        max_epochs=MAX_EPOCHS)

## use the linear model's simplicity to get a sense of the 
## weight assigned to each feature:
fig, ax = plt.subplots(1,1, figsize=(16,9), 
                       gridspec_kw={'left':0.05,'right':0.98,
                                    'top':0.95, 'bottom':0.25})
ax.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
ax.set_xticks(range(len(train_df.columns)))
_ = ax.set_xticklabels(train_df.columns, rotation=90)
ax.set_title("Weights of the linear model for each feature")
ax.set_ylabel("Weight")


"""
    Validation, Performance, Plots and all that
"""
val_performance = {}
performance = {}

val_performance['Baseline'] = baseline.evaluate(wide_window.val)
performance['Baseline'] = baseline.evaluate(wide_window.test, verbose=0)

val_performance['Linear'] = linear.evaluate(wide_window.val)
performance['Linear'] = linear.evaluate(wide_window.test, verbose=0)

val_performance['Simple Dense Model'] = deep_simple.evaluate(wide_window.val)
performance['Simple Dense Model']     = deep_simple.evaluate(wide_window.test, verbose=0)

val_performance['Deep Dense Model'] = deep_complex.evaluate(wide_window.val)
performance['Deep Dense Model']     = deep_complex.evaluate(wide_window.test, verbose=0)

val_performance['Multi-step Dense Model'] = deep_multistep.evaluate(multistep_window.val)
performance['Multi-step Dense Model']     = deep_multistep.evaluate(multistep_window.test, verbose=0)

val_performance['Deep Conv Model'] = deep_conv.evaluate(conv_window.val)
performance['Deep Conv Model']     = deep_conv.evaluate(conv_window.test, verbose=0)

fig1 = wide_window.plot(baseline, max_subplots=3, 
                        title='Model: Baseline', 
                        colors=[input_color,label_color, prediction_color])

fig2 = wide_window.plot(linear  , max_subplots=3, 
                        title='Model: Linear', 
                        colors=[input_color,label_color, prediction_color])

fig3 = wide_window.plot(deep_simple  , max_subplots=3, 
                               title='Model: simple dense model', 
                               colors=[input_color,label_color, prediction_color])

fig4 = wide_window.plot(deep_complex  , max_subplots=3, 
                               title=f'Model: deep, dense model', 
                               colors=[input_color,label_color, prediction_color])

fig5 = multistep_window.plot(deep_multistep, max_subplots=3, 
                             title="Multi-step dense model", 
                             colors=[input_color,label_color, prediction_color])

fig6 = conv_window.plot(deep_conv, max_subplots=3, 
                        title="Multi-step deep conv model", 
                        colors=[input_color,label_color, prediction_color])


"""
    Plot the performances of the models:
"""
## Plot the performance of the various models:
x = np.arange(len(performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = linear.metrics_names.index('mean_absolute_error')

val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

fig, ax = plt.subplots(1,1)
ax.bar(x - 0.17, val_mae, width, label='Validation', color=validation_color)
ax.bar(x + 0.17, test_mae, width, label='Test', color=test_color)
ax.set_xticks(ticks=x, labels=performance.keys(),
           rotation=45)
ax.set_ylabel('MAE (average over all outputs)')
_ = ax.legend()


histories = {'Linear model': hist_linear, 
             'Simple deep model': hist_deep_simple,
             'Complex deep model': hist_deep_complex,
             'Multi-step deep model': hist_deep_multistep,
             'Convolutional Multi-step deep model': hist_deep_conv}

# print(history_linear.params, history_linear.history.keys())
fig2, axes2 = plt.subplots(len(histories), 1, sharex=True)
for ax, model_name in zip(axes2, histories):
    history = histories[model_name]
    ax.plot(history.history['loss'], label='training', color=train_color)
    ax.plot(history.history['val_loss'], label='validation', color=validation_color)
    ax.legend(loc='best')
    ax.text(0.4, 0.85, model_name,transform=ax.transAxes)
axes2[-1].set_xlabel('Epochs')
fig2.supylabel('Loss Function (MSE)')
plt.show()


# # #df_std = (raw_data - train_mean) / train_std
# # train_df = train_df.melt(var_name='Column', value_name='Normalized')
# # plt.figure(figsize=(12, 6))
# # ax = sns.violinplot(x='Column', y='Normalized', data=train_df)
# # _ = ax.set_xticklabels(data.keys(), rotation=90)
# # plt.show()