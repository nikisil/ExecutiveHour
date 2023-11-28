import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd

def random_datetimes_or_dates(start, end, out_format='datetime', n=10): 

    '''   
    unix timestamp is in ns by default. 
    I divide the unix time value by 10**9 to make it seconds (or 24*60*60*10**9 to make it days).
    The corresponding unit variable is passed to the pd.to_datetime function. 
    Values for the (divide_by, unit) pair to select is defined by the out_format parameter.
    for 1 -> out_format='datetime'
    for 2 -> out_format=anything else
    '''
    (divide_by, unit) = (10**9, 's') if out_format=='datetime' else (24*60*60*10**9, 'D')

    start_u = start.value//divide_by
    end_u = end.value//divide_by

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit)

def split_train_test_val_bydate(start_date, end_date, dataset, date_parsing=[], targ_col='date',target_folder='final_data'):
    ### PARAMETERS
    # start_date (string) : start_date for test & validation block (YYYY-MM-DD)
    # end_date (string) : end_date for test & validation block (YYYY-MM-DD)
    # dataset (string) : name of the csv file where entire dataset is stored
    # date_parsing (array): list of columns to be parsed as dates (if more than just the column onto which condition is applied)
    # targ_col (string): name of the column which will be used in loc (to gather dataframe elements which happened before and after start_date)
    ### End PARAMETERS
    if date_parsing:
        dataset = pd.read_csv(dataset,parse_dates=date_parsing).dropna()
    else:
        dataset = pd.read_csv(dataset,parse_dates=[targ_col]).dropna()
    dataset_ordered = dataset.loc[dataset[targ_col] < np.datetime64(start_date)]
    test_val_ordered = dataset.loc[(dataset[targ_col] >= np.datetime64(start_date)) & (dataset[targ_col] <= np.datetime64(end_date))]
    val_date_list = list(random_datetimes_or_dates(pd.to_datetime(start_date),pd.to_datetime(end_date),out_format='not datetime',n=150).values) 
    seasonal_val = test_val_ordered.copy()
    seasonal_val = seasonal_val.loc[seasonal_val[targ_col].isin(val_date_list)]
    dropped_dataset = test_val_ordered.drop(seasonal_val.index)
    seasonal_val.to_csv('%sordered_seasonal_validation_set.csv'%target_folder)
    dropped_dataset.to_csv('%sordered_test_set.csv'%target_folder)
    dataset_ordered.to_csv('%sordered_train_set.csv'%target_folder)

