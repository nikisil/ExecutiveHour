import helpers
import pandas as pd
import numpy as np
#from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from holidays.utils import country_holidays
from datetime import datetime

## create a calendar of the holidays. This date range won't be the 
## final date range of the DataFrame, I just want to be sure to 
## include all the dates. 
#date_range = pd.date_range(start='2019-01-01', end='2024-12-01') 
#cal = calendar()
#holidays = cal.holidays(start=date_range.min(), end=date_range.max())
holidays = country_holidays("US", years=np.arange(2019, 2024, 1))

time_deltas = [1, 2, 19, 20, 21, 22, 23, 24, 25, 26, 49, 168] ## in hours
columns = ['DA_price', 'RT_price', 'load', 'temp', 'dwpt',
             'nat_gas_spot_price', 'monthly_avg_NY_natgas_price']
data_frame = helpers.MainData(read_from_file=False, include_natgas=True, 
                              weather_data_loc='../data/dat_set_3/', 
                              nrg_data_loc='../data/dat_set_3/',
                              holiday_calendar=holidays,
                              time_deltas=time_deltas,
                              columns=columns)
data_frame.process_data()
data_frame.add_previous_day_price_load_averages(day_ahead=False)

df = data_frame.data
df = df[df.time.between(datetime(2010,1,1), datetime(2023,8, 31, 23, 59, 59)) ]


columns_ = df.columns.to_list()
columns_.remove('time')
for v in columns:
    columns_.remove(v)

columns = ['time',*columns, *columns_]
df = df[columns]

index_of_first_non_null = df[~ df.avg_actual_load_prev_day.isna()].index[0]
df = df[index_of_first_non_null:]

df.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)


print(f'There are {df.isnull().values.sum()} missing values or NaNs in the DataFrame.')

df.to_csv("../processed_data/data_set_final_with_average_RT_price.csv", float_format="%0.5f", index=False)