from window_generator import WindowGenerator
import models
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import numpy as np

raw_data = pd.read_csv("../processed_data/data_set_final_with_average_RT_price.csv")
raw_data['time'] = pd.to_datetime(raw_data["time"])
raw_data['date'] = pd.to_datetime(raw_data["date"]).dt.date
raw_data.drop(['RT_price'], inplace=True, axis=1)

"""
    Not all features will be used for the modelling effort
    In this script I will not consider auto-regressive models 
    so choose only the columns that are relevant.
"""
features = ['time','DA_price','load','temp','dwpt',
            'nat_gas_spot_price','monthly_avg_NY_natgas_price',
            'price(h-24)','price(h-25)','price(h-49)','price(h-168)',
            'load(h-24)','load(h-25)','load(h-49)','load(h-168)',
            'hour','weekday','month','day_of_week','holiday',
            'business_hour','season',
            'avg_RT_price_prev_day','avg_actual_load_prev_day']

data = raw_data[features]

time_ = data.pop('time')
## now split the data set to train, validation, test according
## to 70, 20, 10 percent
n = len(data)
train_df = data[0:int(n*0.8)]
val_df   = data[int(n*0.8):int(n*0.95)]
test_df  = data[int(n*0.95):]

## feature scaling:
train_mean = train_df.mean(numeric_only=True, axis=0)
train_std  = train_df.std(numeric_only=True, axis=0)

train_df = (train_df - train_mean) / train_std
val_df   = (val_df   - train_mean) / train_std
test_df  = (test_df  - train_mean) / train_std



column_indices = {name: i for i, name in enumerate(data.columns)}
window_gen = WindowGenerator(input_width=24, 
                             label_width=24, 
                             shift=24, 
                             train_df=train_df, val_df=val_df,
                             test_df=test_df, label_columns=['DA_price'])

for example_inputs, example_labels in window_gen.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


baseline = models.Baseline(label_index=column_indices['DA_price'])

baseline.compile(loss=keras.losses.MeanSquaredError(),
                 metrics=[keras.metrics.MeanAbsoluteError()])

linear = models.Linear()
history_linear = models.compile_and_fit(linear, window=window_gen, 
                                        patience=10, max_epochs=100)

nlayers = 4
deepNN = models.FullyConnectedDense(units=24, layers=nlayers)
history_deepNN = models.compile_and_fit(deepNN, window=window_gen,
                                        patience=10, max_epochs=100,
                                        learning_rate=0.0005)

val_performance = {}
performance = {}

val_performance['Baseline'] = baseline.evaluate(window_gen.val)
performance['Baseline'] = baseline.evaluate(window_gen.test, verbose=0)

val_performance['Linear'] = linear.evaluate(window_gen.val)
performance['Linear'] = linear.evaluate(window_gen.test, verbose=0)

val_performance['deepNN'] = deepNN.evaluate(window_gen.val)
performance['deepNN'] = deepNN.evaluate(window_gen.test, verbose=0)

window_gen.plot(baseline, max_subplots=3, title='Model: Baseline')
window_gen.plot(linear  , max_subplots=3, title='Model: Linear')
window_gen.plot(deepNN  , max_subplots=3, title=f'Model: deepNN with {nlayers} layers')

print("Validation: ")
for v, p in val_performance.items():
    print(v, p)

print("Performance: ")
for v, p in performance.items():
    print(v, p)


## Plot the performance of the various models:
x = np.arange(len(performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = deepNN.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

fig, ax = plt.subplots(1,1)
ax.bar(x - 0.17, val_mae, width, label='Validation')
ax.bar(x + 0.17, test_mae, width, label='Test')
ax.set_xticks(ticks=x, labels=performance.keys(),
           rotation=45)
ax.set_ylabel('MAE (average over all outputs)')
_ = ax.legend()


histories = {'Linear': history_linear, 'DeepNN':history_deepNN}
print(history_linear.params, history_linear.history.keys())
fig2, axes2 = plt.subplots(len(histories), 1, sharex=True)
for ax, model_name in zip(axes2, histories):
    history = histories[model_name]
    ax.plot(history.history['loss'], label='training')
    ax.plot(history.history['val_loss'], label='validation')
    ax.legend(loc='best')
    ax.text(0.4, 0.1, model_name,transform=ax.transAxes)
plt.show()


# #df_std = (raw_data - train_mean) / train_std
# train_df = train_df.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=train_df)
# _ = ax.set_xticklabels(data.keys(), rotation=90)
# plt.show()

# plt.bar(x = range(len(train_df.columns)),
#         height=linear.layers[0].kernel[:,0].numpy())
# axis = plt.gca()
# axis.set_xticks(range(len(train_df.columns)))
# _ = axis.set_xticklabels(train_df.columns, rotation=90)
# plt.show()