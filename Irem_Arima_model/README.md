**ARIMA Model**

This folder contains the ARIMA model built for Day-ahead (DA) electricity price prediction in the New York area. The base electricity data is scraped from NRG website. In addition to the electricity prices, the data also includes the weather and natural gas price information. The dataset contains hourly information from 10-27-2020 to 10-01-2023. 

The model uses the training, validation, and test sets that Nic has prepared using the raw data above. After some data analysis, based on the Pearson correlation coefficients with the DA prices and some grid search the following features have been chosen for the final model: 
`DA_price(t-1D)` , `avg_DA_price (h-24)`, `DA_price(t-2D)` , `DA_price(t-3D)`, `DA_price(t-4D)`, 
`DA_price(t-5D)`, where `DA_price(t-kD)` is the day-ahead price of `k`-days ago. 

The model with the following hyperparameters has been trained with the training data: 

$\mathrm{ARIMA} = \mathrm{SARIMAX}(\mathrm{endog}=\mathrm{train_data}[\mathrm{'DA_price'}],\mathrm{exog}=\mathrm{train_data}[f_{1}], \mathrm{order}=\left(24, 1, 7\right)).\mathrm{fit}(\mathrm{maxiter}=200)$  
 where $f_{1}$ is the Python list of the features listed above. These features were fed into the model as additional features that have an effect on the hourly DA prices. 
Since the DA prices up to 24 hours prior have high correlations with the current day DA prices, I chose the autoregressive model of order 24 to regress my model onto the observations of the previous 24 hours. Due to the nonstationary and noisy nature of the data, 1-time differencing and a relatively big parameter for the moving-average part of Arima were needed. For these reasons, the order in the Arima model was chosen as $(24, 1, 7)$. 

RMSE of the model on the validation and test sets is: 7.198 and 7.213, respectively. 

Results show that the Arima model does a decent job forecasting DA prices compared to the baseline model where the prediction for the current DA price is the DA_price (t-1D). The downside of the Arima model on the other hand is that depending on the maximum iteration number, the training may take a long time.

In this folder, one can find a sample Jupyter notebook demonstrating how to use the ArimaModel() class with a trivial example and another Jupyter notebook that has the code of the same model with annotations. 
