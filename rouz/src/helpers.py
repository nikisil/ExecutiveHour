import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class NRGData:
    def __init__(self, location:str = "../data/dat_set_3/") -> None:

        ## read the data:
        self._tmp_da = pd.read_csv(location + "market_data_day_ahead.csv")
        self._tmp_rt = pd.read_csv(location + "market_data_real_time.csv")
        self._tmp_ld = pd.read_csv(location + "market_data_actual_load.csv")
        self._data = None

    def process_nrg_data(self):
        ## now process and join the data:
        self._tmp_da.drop(['label','type'], inplace=True, axis=1)
        self._tmp_rt.drop(['label','type'], inplace=True, axis=1)
        self._tmp_ld.drop(['label','type'], inplace=True, axis=1)

        for item in [self._tmp_da, self._tmp_rt, self._tmp_ld]:
            item.time = pd.to_datetime(item.time, utc=True)
            item.time = item.time.dt.tz_localize(None)

        self._tmp_da.rename(columns={'value':'DA_price'}, inplace=True)
        self._tmp_rt.rename(columns={'value':'RT_price'}, inplace=True)
        self._tmp_ld.rename(columns={'value':'load'}, inplace=True)

        tmp_df = self._tmp_da.merge(self._tmp_rt, on=['time'], how='inner')
        self._data = tmp_df.merge(self._tmp_ld, on=['time'], how='inner')

    @property
    def data(self):
        return self._data


class WeatherData:

    def __init__(self, loc:str = "../data/dat_set_3/") -> None:
        self._data = pd.read_csv(loc + "NYC_weather_data.csv", parse_dates=['dtime']) ## weather data
        self._data.drop(['index','week','month'], inplace=True, axis=1)
        self._data['time'] = self._data['dtime'].dt.tz_convert('UTC')
        self._data['time'] = self._data['time'].dt.tz_localize(None)
        self._data.drop(['rhum', 'prcp', 'wdir', 'wspd', 'pres', 'coco', 'weekly_T_anom',
       'monthly_T_anom', 'weekly_Prec_anom', 'monthly_Prec_anom',
       'weekly_Wind_anom', 'monthly_Wind_anom', 'weekly_Pressure_anom',
       'monthly_Pressure_anom', 'snowing', 'raining', 'hail', 'cloudy','dtime'], inplace=True, axis=1)

    @property
    def data(self):
        return self._data
    
    def merge(self, other, **kwargs):
        tmp = self._data.merge(other, **kwargs)
        return tmp


class MainData:
    def __init__(self, read_from_file=False, 
                 pre_processed_data_fname='./tmp.csv', 
                 include_natgas = False,
                 natgas_fname = "../../clean_dat_LSTM_natgas.csv",
                 weather_data_loc = './',
                 nrg_data_loc = './',
                 holiday_calendar=None,
                 time_deltas = None,
                 columns=None):
        self.read_file = read_from_file
        self.loc = pre_processed_data_fname
        self._data = None
        self._weather_loc = weather_data_loc
        self._weather = None
        self._nrg_loc = nrg_data_loc
        self._holidays = holiday_calendar
        self._time_deltas = time_deltas
        self._include_natgas = include_natgas
        self._natgas_fname = natgas_fname
        self._columns = columns

    @property
    def data(self):
        return self._data

    def _include_nrg_data(self):
        self._nrg_data_ = NRGData(self._nrg_loc)
        self._nrg_data_.process_nrg_data()

    def _include_weather_data(self):
        self._weather = WeatherData(self._weather_loc)
    
    def _merge_weather_and_nrg(self):
        self._data = self._weather.merge(self._nrg_data_.data)
        self._data.reset_index(drop=True, inplace=True)
        self._data.drop_duplicates(keep='last', subset=['time'], inplace=True)

    def _include_nat_gas_prices(self):
        ngas_df = pd.read_csv(self._natgas_fname)
        ngas_df = ngas_df[['time','nat_gas_spot_price', 'monthly_avg_NY_natgas_price']]
        ngas_df['time'] = pd.to_datetime(ngas_df['time'])
        ngas_df.drop_duplicates(keep='last', subset=['time'], inplace=True)
        self._data = self._data.merge(ngas_df, on=['time'], how='inner', validate='one_to_one')

    def process_data(self):
        if self.read_file:
            format_string = "%Y-%m-%d %H:%M:%S"
            self._data = pd.read_csv(self.loc)
            self._data['time'] = [datetime.strptime(v, format_string) for v in self._data.time]
            self._data.date = [datetime.strptime(v, "%Y-%m-%d").date() for v in self._data.date]
        else:
            self._include_nrg_data()
            self._include_weather_data()
            self._merge_weather_and_nrg()
            self._include_nat_gas_prices()
            self.create_price_and_load_features()
            self.expand_time_feature()
            print("Processed the various data files.")
            print(f"We now have the dataframe:\n {self._data.info()}")
            print(''.join(['**'*10]))

    def expand_time_feature(self):
        print("Expand time featuer to new columns of day, week, date, hour ...")
        df = self._data
        df['date']          = df['time'].dt.date
        df['hour']          = df['time'].dt.strftime("%H").astype('int')
        df['weekday']       = df['time'].dt.dayofweek <= 4
        df['month']         = df['time'].dt.month
        df['month']         = df['month'].astype('int')
        df['day_of_week']   = df['time'].dt.dayofweek + 1
        df['holiday']       = [(v in self._holidays) for v in df.date]
        df['business_hour'] = (df['weekday'].astype(bool)) & (df['hour'].between(8, 17))
        for col in ['hour', 'weekday', 'business_hour', 'holiday']:
            df[col].astype(np.int64)
        df.replace({False: 0, True: 1}, inplace=True)

        seasons = []
        for month in df.month.values:
            season = None
            if month in [1,2,12]:
                season = 0
            elif month in [3,4,5]:
                season = 1
            elif month in [6,7,8]:
                season = 2
            else:
                season = 3
            seasons.append(season)

        df['season'] = seasons
        self._data = df

    def create_price_and_load_features(self):
        """
            check if for each given time t, t-(times) are also present
            in the data frame
        """
        print("Create new features of day ahead price and load features at specified hours...")
        df = self._data
        times = self._time_deltas
        columns = self._columns

        first_entry = df.time.iloc[0]
        missing_times = {}
        df_times = df.time.values
        for dt in times:
            time_delta = timedelta(hours=dt)
            for _, row in df.iterrows():
                curr_time = row.time
                past_time = curr_time - time_delta
                if (past_time not in df_times and 
                    past_time not in missing_times):
                    #print(f"missing {past_time}")
                    missing_times[past_time] = {col : np.nan for col in columns} 
                    missing_times[past_time]['time'] = past_time
                    #missing_times[past_time]['date'] = past_time.date()
        print(f"Missing {len(missing_times)} times." )
        ## turn missing_times into a dataframe and concatenate it with the original
        missing = pd.DataFrame(missing_times.values())

        tmp = pd.concat([df, missing])
        tmp.sort_values(by='time', ascending=True, inplace=True)
        tmp.reset_index(inplace=True, drop=True)
        ## now go through and add in the prices/loads
        for index, row in tmp.iterrows():
            for dt in times:
                if row.time < first_entry:
                    tmp.at[index, f"price(h-{dt})"] = np.nan
                    tmp.at[index, f"load(h-{dt})"]  = np.nan
                    continue
                time_delta = timedelta(hours=dt)
                relevant_time = row.time - time_delta
                relevant_row = tmp[tmp.time == relevant_time]

                tmp.at[index, f"price(h-{dt})"] = relevant_row.iloc[0]['DA_price']            
                tmp.at[index, f"load(h-{dt})"] = relevant_row.iloc[0]['load']

        self._data = tmp[tmp.time >= first_entry]

    def add_previous_day_price_load_averages(self, day_ahead=False):
        """
            Add the average load and price of the previous day
            by default adds the average real time price of the 
            day before. 
            Day is the calendar day before. Thus the average is 
            not a _rolling_ average of the previous 24 hours.
        """
        df = self._data
        label = 'DA_price' if day_ahead else 'RT_price'
        indx_init = df[df.hour == 0].index[0] ## first hour 0 

        period = 24
        avg_price = df.iloc[indx_init : indx_init+period][label].mean()
        avg_load  = df.iloc[indx_init : indx_init+period]['load'].mean()

        yesterday = df.iloc[indx_init]['time'].date()

        col_mean_price, col_mean_load = f"avg_{label}_prev_day", "avg_actual_load_prev_day"
        df[col_mean_price] = np.nan
        df[col_mean_load] = np.nan
        index_col_mean_price = df.columns.get_loc(col_mean_price)
        index_col_mean_load = df.columns.get_loc(col_mean_load)

        dframe_size = len(df)
        indx = indx_init+period
        dt_day = timedelta(days=1)
        dt_hour = timedelta(hours=1)

        while indx < dframe_size:
            today = df.iloc[indx].time.date()
            #print(f"{indx} for today:{today}, using data of yesterday:{yesterday}")
            tmp_indx = indx
            #while (today - dt_day == yesterday):
            for _ in range(24):
                #print(f"Today: {today}, Yesterday: {yesterday}, Time Delta: {today-yesterday} ")
                if tmp_indx >= dframe_size:
                    break
                df.iat[tmp_indx, index_col_mean_price] = avg_price 
                df.iat[tmp_indx, index_col_mean_load]  = avg_load
                # today = df.iloc[tmp_indx].time.date()
                today += dt_hour
                tmp_indx += 1

            ## no longer on the same day, so update averages & the index:
            if indx + period > dframe_size:
                break
            avg_price = df.iloc[indx : indx+period][label].mean()
            avg_load  = df.iloc[indx : indx+period]['load'].mean()
            indx += period
            yesterday = today
        self._data = df