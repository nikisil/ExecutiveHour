import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SARIMAX
import matplotlib.dates as mdates

#need train-test data sets to be read_csv with parse_dates = ['time']
#the training set the original model uses is: ordered_train_set.csv
#the test set is: full_ordered_test_set.csv 
#Each of these csv files are in ExecutiveHour/data_processing/final_data



class ArimaModel():
    def __init__(self, train_df, test_df):  
        
        self.training_data = train_df.set_index('time')
        self.training_data.index = pd.DatetimeIndex(self.training_data.index).to_period('H')
        
        self.test_data = test_df.set_index('time')
        self.test_data.index = pd.DatetimeIndex(self.test_data.index).to_period('H')
        
        self.arima = None
        self.exog_features = None
        self.predictions = pd.DataFrame()
        self.RMSE = 0
        
    def train(self, order=(24, 1, 7), maxiter = 200, exog_features = None):
        
        if exog_features == None:
            
            self.exog_features = ['price(h-25)','price(h-24)','price(h-26)',
                                  'price(h-49)','DA_price(t-2D)','DA_price(t-3D)']
        else:
            
            self.exog_features = exog_features

                
        self.arima = SARIMAX(endog=self.training_data['DA_price'],
                            exog=self.training_data[self.exog_features], order=order).fit(maxiter=maxiter)
            
            
    def get_preds(self):
        
        if self.arima == None:
            raise Exception('ARIMA has not been trained yet.')
            
        else:
            
            a = int(len(self.test_data)/24)
            
            dailyRMSE = 0
            squared_errors = 0

            
            for i in range(a):
                
                if i == 0:
                    forecasts = self.arima.forecast(steps=24, exog=self.test_data[self.exog_features][:24])
                    squared_errors += ((forecasts - self.test_data['DA_price'][:24])**2).sum()
                    forecasts = forecasts.set_axis(forecasts.index.hour)
                    dailyRMSE = np.sqrt(mean_squared_error(forecasts,self.test_data['DA_price'][:24]))
                    ser = pd.Series([dailyRMSE]).set_axis(['DailyRMSE'])
                    self.predictions[f'Day {i+1} fcast'] = pd.concat([forecasts, ser])
                    
                else:
                    new_ob = self.arima.append(endog=self.test_data['DA_price'][:i*24], 
                                               exog=self.test_data[self.exog_features][:i*24]) 
                    forecasts = new_ob.forecast(steps=24, exog=self.test_data[self.exog_features][i*24:(i+1)*24])               
                    squared_errors += ((forecasts - self.test_data['DA_price'][i*24:(i+1)*24])**2).sum()
                    forecasts = forecasts.set_axis(forecasts.index.hour)
                    dailyRMSE = np.sqrt(mean_squared_error(forecasts,self.test_data['DA_price'][i*24:(i+1)*24]))
                    ser = pd.Series([dailyRMSE]).set_axis(['DailyRMSE'])
                    self.predictions[f'Day {i+1} fcast'] = pd.concat([forecasts, ser])
                    
            self.RMSE = np.sqrt(squared_errors/len(self.test_data))
            
            return self.predictions
        
     


    def sample_plot(self):
        
        fig, (ax1, ax2) = plt.subplots(2)
        fig.set_figwidth(10)
        fig.set_figheight(5)
        fig.set_dpi(200)
        ax1.plot(self.training_data.index.to_timestamp()[-120:], 
                 self.training_data['DA_price'][-120:],
                'r-',
                label= 'Last 5 days of training data')
        ax1.plot(self.training_data.index.to_timestamp()[-120:], 
                 self.arima.fittedvalues[-120:],
                'g--',
                label= 'Arima(24,1,7) fit on training data')

        ax2.plot(self.test_data.index.to_timestamp()[:120],
                 self.test_data['DA_price'][:120],
                    'b-',
                  label="First 5 days of test data")
        
        ax2.plot(self.test_data.index.to_timestamp()[:24],
                 self.predictions.drop('DailyRMSE',axis=0)['Day 1 fcast'],
                    'g--*',
                    label="ARIMA forecast on the first 5 days of test data")
        
        for i in range(1,5):
            ax2.plot(self.test_data.index.to_timestamp()[i*24:(i+1)*24],
                 self.predictions.drop('DailyRMSE',axis=0)[f'Day {i+1} fcast'],
                    'g--*',
                    )

        ax1.set_ylabel('DA Price')
        ax2.set_ylabel('DA Price')
        ax2.set_xlabel('Time')
        ax1.legend(fontsize=8, loc='best')
        ax2.legend(fontsize=8, loc='best')
        return fig, (ax1, ax2)
        

