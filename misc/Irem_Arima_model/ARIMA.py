import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SARIMAX


#need train-val-test data sets to be read_csv with parse_dates = ['time']
#the training set the original model uses is: ordered_train_set.csv
#the validation set is: ordered_seasonal_validation_set.csv 
#the test set is: ordered_test_set.csv 
#Each of these csv files are in https://github.com/nikisil/ExecutiveHour/tree/Main/final_datasets


class ArimaModel():
    def __init__(self, train_df, val_df, test_df):
        
        self.training_data = train_df.set_index('time')
        self.training_data.index = pd.DatetimeIndex(self.training_data.index).to_period('H')
        
        self.validation_data = val_df.set_index('time')
        self.validation_data.index = pd.DatetimeIndex(self.validation_data.index).to_period('H')
        
        self.test_data = test_df.set_index('time')
        self.test_data.index = pd.DatetimeIndex(self.test_data.index).to_period('H')
        
        self.arima = None
        self.exog_features = None
        self.validation_predictions = None
        self.test_predictions = None
        
    def train_arima(self, order=(24, 1, 7), maxiter = 200, exog_features = None):
        
        if exog_features == None:
            
            self.exog_features = ['DA_price(t-1D)', 'DA_price(t-2D)','DA_price(t-3D)', 
                         'DA_price(t-4D)', 'DA_price(t-5D)', 'avg_DA_price(h-24)']
        else:
            self.exog_features = exog_features
            
        
        self.arima = SARIMAX(endog=self.training_data['DA_price'],
                            exog=self.training_data[self.exog_features], order=order).fit(maxiter=maxiter)
            
            
    def get_validation_rmse(self):
        if self.arima == None:
            raise Exception('ARIMA has not been trained yet.')
        else:
            val_obs = self.arima.apply(endog = self.validation_data['DA_price'], 
                                       exog=self.validation_data[self.exog_features], refit=False)
            
            self.validation_predictions = val_obs.predict(0,len(self.validation_data)-1, dynamic=False)
            
            return np.sqrt(mean_squared_error(self.validation_data['DA_price'],self.validation_predictions))
    
    def get_test_rmse(self):
        
        if self.arima == None:
            raise Exception('ARIMA has not been trained yet.')
        else:
            test_obs = self.arima.apply(endog = self.test_data['DA_price'], 
                                       exog=self.test_data[self.exog_features], refit=False)
            
            self.test_predictions = test_obs.predict(0,len(self.test_data)-1, dynamic=False)
            
            return np.sqrt(mean_squared_error(self.test_data['DA_price'],self.test_predictions))
            