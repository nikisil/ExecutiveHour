XGBoost Model for Day Ahead price forecasting in New York City. 

This folder contains the following files:
    - xgboost-11-27-2023.ipynb
    - XGB.py
    - sample_use.ipynb

The xgboost-11-27-2023.ipynb Jupyter notebook contains an annotated description of building up the xgboost model. It also contains the predictions of this model on the validation and test data sets, along with comparisons with other basic models such as LinearRegression and our baseline models (which is the the Day Ahead price from 24 hours prior). 

The XGB.py file contains the class XGBModel. 
    - Initialization of model: To initialize the XGBModel, we feed it the training, validation, test data, features, and target. Default values are provided for features and the target. The default value for the features consist of those in xgboost-11-27-2023.ipynb. The default value for target is 'DA_price'. 

    - XGBModel.train(): Trains the model with some hyperparameters determined by experimentation. 

    - XGBModel.feature_importance(): Gives a barplot of the feature importance in the construction of the XGBoost model.

    - XGBModel.predict( pred_on = 'custom', pred_date = '', pred_df = pd.DataFrame()): Returns triple (A,b,c), where:
        * 'A' is a data frame with the actual Day Ahead price, XGBoost's predicted Day Ahead price, and baseline Day Ahead price
        * 'b' is the RMSE of XGBoost's predicted Day Ahead price with the actual Day Ahead price
        * 'c' is the RMSE of the baseline Day Ahead price with the actual Day Ahead price
    
    Input parameters: 
        * 'pred_on' can be set to 'validation', 'test', or 'custom' (it is 'custom' by default). If set to 'validation' or 'test', this function returns the predictions for the validation or test dataframes, respectively. If set to 'custom', the function returns the predictions on a custom input dataframe, 'pred_df'. 

        * 'pred_df' can be set to any dataframe with formatting matching that of the train, validation, and test data sets (it is set to the empty dataframe by default). For example, we can set pred_df to today's data (obtained from sources such as the NRG website) to get predictions for tomorrow's prices. 

        * 'pred_date' can be set to a date of the form 'YYYY-MM-DD' (it is set to the empty string by default). If 'pred_date' is provided, the function checks if 'pred_date' is in the set of the dates contained in 'pred_df'. And in such a case, the function returns the XGBoost predictions just for that date. 

        * Example uses: 
            -- XGBModel.predict(pred_on = 'validation')
            -- XGBModel.predict(pred_on = 'test', pred_date = '2023-07-25')
            -- XGBModel.predict(pred_df = current_week_data, pred_date = '2023-11-28')


The sample_use.ipynb contains an example of how to use the XGBModel class.  
        

      