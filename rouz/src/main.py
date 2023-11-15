from window_generator import WindowGenerator
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv("../rouz_final_DF_v2.csv")
raw_data['time'] = pd.to_datetime(raw_data["time"])
raw_data['date'] = pd.to_datetime(raw_data['date']).dt.date


raw_data = raw_data.loc[:,~raw_data.columns.str.match("Unnamed")]
## first day in the data is 2020-10-27. Remove the first week to avoid 
## negative numbers
one_week  = timedelta(weeks=1)
lower_cut = datetime.fromisoformat('2020-10-27') + one_week
upper_cut = datetime.fromisoformat('2023-09-30')
raw_data = raw_data[raw_data.date.between(lower_cut.date(), upper_cut.date())]
raw_data.pop('date')
time_ = raw_data.pop('time')
## now split the data set to train, validation, test according
## to 70, 20, 10 percent
n = len(raw_data)
train_df = raw_data[0:int(n*0.7)]
val_df   = raw_data[int(n*0.7):int(n*0.9)]
test_df  = raw_data[int(n*0.9):]

num_features = raw_data.shape[1]

print(f"length of raw data: {n}, num. features: {num_features}")

## feature scaling:
train_mean = train_df.mean(numeric_only=True, axis=0)
train_std  = train_df.std(numeric_only=True, axis=0)

train_df = (train_df - train_mean) / train_std
val_df   = (val_df - train_mean)   / train_std
test_df  = (test_df - train_mean)  / train_std

#df_std = (raw_data - train_mean) / train_std
train_df = train_df.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=train_df)
_ = ax.set_xticklabels(raw_data.keys(), rotation=90)

plt.show()