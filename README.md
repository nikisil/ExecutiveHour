# ExecutiveHour: Energy-price forecasting

by Irem Altimer, Nicolas Fortier, Souparna Purohit & Rouzbeh Modarresi Yazdi

Electricity and energy markets are an integral part of the electrical grid, allowing for the matching of the supply and demand of electricity via market mechanisms. Accurate forecasting of electricity demand and price is crucial given that generated power needs to be consumed immediately and it cannot be stored. In this work, we focus on the day-ahead markets for electricity, where participants purchase or supply electricity to be delivered the following day. This mechanism accounts for $95$\% of overall transactions (NRG Energy, Inc.) and avoids the volatility of the real-time price.  

## Target market and data step
Our chosen market was determined to be the New York City wholesale electricity market, allowing for focus on a relatively small geographical area. We considered natural factors such as electricity demand and local weather conditions. The initial data set was constructed from the hourly day-ahead prices, grid-load data and local weather information (temperature, dew-point etc).  

The energy price data (both real-time and day-ahead) as well as electricity load data was sourced from the [NRG](https://www.nrg.com/resources/energy-tools/tracking-the-market.html). The NRG corporation provides free access to this data going back three years. Our final data set then contains load and price data from 27th of October 2020 to the 1st of October 2023.  

This was augmented, based on a preliminary analysis of the time series, by price data at a certain number of previous time steps (in hours). The feature set was further expanded to include categorical temporal information (day of the week, weekday vs weekend etc), as well as monthly average natural gas prices in the country. For more information on the data preprocessing and merging, see the [data processing notebook](./data_processing/data_processing.ipynb) as well as the Quick Walkthrough provided below. 

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

## Setup

If you would like to reproduce (or build upon) the results and models developed in this analysis, you will need the following python packages:

- jupyter
- pandas
- numpy
- matplotlib
- tensorflow
- keras
- holidays
- xgboost
- statsmodels
- sklearn
- meteostat

along with all default packages included in Python 3.10. Using an older version of python may lead to compatibility issues.

## Quick Walkthrough

### Data-Processing

Our work is divided into annotated notebooks. To start, one could want to regenerate (or add to) our data. To do so, go to the [data_processing](./data_processing/data_processing.ipynb) Jupyter Notebook. This notebook makes use of functions and classes that were developed specifically for the problem at hand and contained in [helpers](./data_processing/helpers.py), [preprocess](./data_processing/preprocess.py), [train_test_validate_split](./data_processing/train_test_validate_split.py) & [weather](./data_processing/weather.py). The [data processing notebook](./data_processing/data_processing.ipynb) relies on [raw energy data](./data_processing/raw_data/dat_set_3/) (pulled from [NRG](https://www.nrg.com/resources/energy-tools/tracking-the-market.html)), as well as [average natural gas prices](./data_processing/raw_data/nat_gas_prices.csv) (pulled from [EIA](https://www.eia.gov/dnav/ng/hist/n3035ny3m.htm)). It pulls weather data using the [meteostat](https://dev.meteostat.net/python/). All of these data sources are then merged into a single dataframe in the [data processing](./data_processing/data_processing.ipynb), which creates train, validation and test sets, which it saves in [final_data](./data_processing/final_data/).

### Training and Running Models

Once all of the data is cleaned and generated, we move to the [final](./final/) directory. Here, two model types are provided: Simple [single step models](./final/single_step_models/), which were trained to predict Day-Ahead prices one hour into the future given varying amounts of information, and [multi step models](./final/multi_step_models/), which were trained to predict the next 24 hours of Day-Ahead prices all at once. Both of these sub-directories contain notebooks ([single](./final/single_step_models/ExecutiveHour.ipynb) and [multi](./final/multi_step_models/ExecutiveHour.ipynb)) which can be used to train, test and modify models as need be. All of the models are developed in python scripts provided in the same directories as their corresponding Jupyter notebooks.

### Other directories

[Final datasets](./final_datasets/) contains final train, validation and test datasets used in training. These are recorded in this specific directory so that any re-run of the [data processing](./data_processing/data_processing.ipynb) notebook will not affect these sets (and therefore preserve their state for posterity). The [miscellaneous](./misc) directory contains various notebooks, scripts (which may or may not be found in the [final](./final/) and [data processing](./data_processing/) directories) and exogenous datasets which were used at some point or another in our exploration. Note that code in the [misc](./misc/) directory may not be as well documented as code found elsewhere in the git, so any questions regarding files contained within said directory can be directed to team members.

## Conclusion  

This exploratory analysis has shown that models using simple time-forecasting features can beat a robust baseline model. The Neural Networks and Boosted Decision tree models could benefit from more precise exogenous data, such as actual industrial natural gas prices in the region, renewable energy generation history (and capacity), as well as load predictions provided by energy providers. These models could then be expanded to provide industrial actors insight into the advantage (or disadvantage) of buying energy in advance as opposed to buying it in the real-time energy market. Without exhaustive exogenous data, however, it is clear that most models, as different as they may be, only outperform our baseline predictor by $\approx 5$\%.
