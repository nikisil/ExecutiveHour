import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class XGBModel():
    def __init__(self, train_df, validation_df, test_df, features = [], target = 'DA_price'):
        self.df_train = train_df
        self.df_validation = validation_df
        self.df_test = test_df
        self.features = features

        if len(self.features) == 0:
            self.features = ['load', 'hour',
                            'month', 'day_of_week', 'holiday', 'business_hour', 'season', 'temp',
                            'dwpt', 'avg_RT_price_prev_day', 'avg_actual_load_prev_day',
                            'RT_price(t-1D)', 'DA_price(t-1D)', 'load(t-1D)', 'RT_price(t-2D)',
                            'DA_price(t-2D)', 'load(t-2D)', 'RT_price(t-3D)', 'DA_price(t-3D)',
                            'load(t-3D)', 'RT_price(t-4D)', 'DA_price(t-4D)', 'load(t-4D)',
                            'RT_price(t-5D)', 'DA_price(t-5D)', 'load(t-5D)', 'RT_price(t-6D)',
                            'DA_price(t-6D)', 'load(t-6D)', 'RT_price(t-7D)', 'DA_price(t-7D)',
                            'load(t-7D)', 'nat_gas_spot_price', 'monthly_avg_NY_natgas_price']

        self.target = target
        self.X_train = self.df_train[self.features]
        self.y_train = self.df_train[self.target]
        self.X_validation = self.df_validation[self.features]
        self.y_validation = self.df_validation[self.target]

        self.model = None

    def train(self):
        self.model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                                    n_estimators=1000,
                                    objective='reg:linear',
                                    early_stopping_rounds=100,
                                    max_depth=3,
                                    learning_rate=0.01)      
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_train, self.y_train), (self.X_validation, self.y_validation)],
              verbose=100)
        
    def feature_importance(self):
        if self.model == None:
            print("Call train() first!")
            return 
        
        fi = pd.DataFrame(data = self.model.feature_importances_,
                          index = self.model.feature_names_in_,
                          columns=['Importance'])
        fi.sort_values('Importance').plot(kind='barh', title='Feature Importance')
        plt.show()

    def predict(self, pred_on = 'custom', pred_df = pd.DataFrame(), pred_date = ''):
        if self.model == None:
            print("Call train() first!")
            return 
        
        if pred_on == 'validation':
            print('Overriding pred_df with df_validation!')
            pred_df = self.df_validation
        
        elif pred_on == 'test':
            print('Overriding pred_df with df_test!')
            pred_df = self.df_test
        
        if pred_date != '':
            if pred_df.empty:
                raise Exception('pred_df is empty!')

            if pred_date not in pred_df['date'].values:
                raise Exception("Invalid date!")
            
            else:
                print('Overriding pred_df with date provided!')
                pred_df = pred_df.loc[ pred_df['date'] == pred_date ]

        if not pred_df.empty:
            X_pred_df = pred_df[self.features]
            custom_pred = self.model.predict(X_pred_df)
            predictions = pred_df[['time', 'DA_price', 'DA_price(t-1D)']]
            predictions.reset_index(drop=True, inplace=True)
            predictions = predictions.rename(columns = { 'DA_price(t-1D)' : 'Baseline_predictions' })
            predictions['XGBoost_predictions'] = custom_pred

            xgb_rmse = np.sqrt(mean_squared_error(predictions['DA_price'], predictions['XGBoost_predictions']))
            baseline_rmse = np.sqrt(mean_squared_error(predictions['DA_price'], predictions['Baseline_predictions']))

            return predictions, xgb_rmse, baseline_rmse
        
        else:
            raise Exception('Invalid input. Something is wrong!')

        
        
        


