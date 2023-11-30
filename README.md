# ExecutiveHour: Energy-price forecasting
Electricity and energy markets are an integral part of the electrical grid, allowing for the matching of the supply and demand of electricity via market mechanisms. Accurate forecasting of electricity demand and price is crucial given that generated power needs to be consumed immediately and it cannot be stored. In this work, we focus on the day-ahead markets for electricity, where participants purchase or supply electricity to be delivered the following day. This mechanism accounts for $95\%$ of overall transactions (NRG Energy, Inc.) and avoids the volatility of the real-time price.  

## Target market and data step
Our chosen market was determined to be the New York City wholesale electricity market, allowing for focus on a relatively small geographical area. We considered natural factors such as electricity demand and local weather conditions. The initial data set was constructed from the hourly day-ahead prices, grid-load data and local weather information (temperature, dew-point etc).  

The energy price data (both real-time and day-ahead) as well as electricity load data was sourced from the [NRG](https://www.nrg.com/resources/energy-tools/tracking-the-market.html). The NRG corporation provides free access to this data going back three years. Our final data set then contains load and price data from 27th of October 2020 to the 1st of October 2023.  

This was augmented, based on a preliminary analysis of the time series, by price data at a certain number of previous time steps (in hours). The feature set was further expanded to include categorical temporal information (day of the week, weekday vs weekend etc), as well as monthly average natural gas prices in the country. For more information on the data preprocessing and merging, see notebook in link.

## Modelling  

Motivated by the preprocessing stage, the baseline was chosen to be a simple model where the predicted values for day-ahead price in the next 24 hours is the day-ahead price of the previous 24 hours. From here the team pursued different model classes: dense neural networks, convolutional networks, LSTMs, boosted decision trees and an autoregressive integrated moving average (ARIMA) model. The overall performance of the different models are quite close to each other, with convolutional and dense models doing marginally better than the others. The associated RMSE for these models, approximately $(20, 20)$ USD/MWh for validation and test sets respectively, is slightly lower than the baseline results of $(21, 20.4)$ USD/MWh. The values are provided in the table below. 

|         Model | Validation | Test  |
|--------------:|-----------:|-------|
|      Baseline |      21.22 | 20.42 |
|        Linear |      20.20 | 20.14 |
|         Dense |      19.39 | 19.32 |
|           CNN |      20.13 | 19.92 |
|          LSTM |      24.96 | 28.25 |
| XGB Regressor |      20.72 | 22.01 |
|         ARIMA |       0.00 | 20.65 |

The models are presented in this [notebook](./final/multi_step_models/ExecutiveHour.ipynb).

## Conclusion  

This exploratory analysis has shown that models using simple time-forecasting features can beat a robust baseline model. The Neural Networks and Boosted Decision tree models could benefit from more precise exogenous data, such as actual industrial natural gas prices in the region, renewable energy generation history (and capacity), as well as load predictions provided by energy providers. These models could then be expanded to provide industrial actors insight into the advantage (or disadvantage) of buying energy in advance as opposed to buying it in the real-time energy market. Without exhaustive exogenous data, however, it is clear that most models, as different as they may be, only outperform our baseline predictor by $\approx 5\%$.
