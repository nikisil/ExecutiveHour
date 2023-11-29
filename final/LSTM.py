import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import tensorflow as tf
import sys

def baseline(data,pred_date):

    # data is the pandas dataframe being used (this assumes we are using rouz's format)
    # pred_date is a string, which can be written as ‘YYYY-MM-DD’ or ‘YYYY-MM-DDTHH:MinMin'

    prev_day = np.datetime64(pred_date) - np.timedelta64(1,'D')
    index = data.loc[data['time'] == prev_day].index.to_list()[0]
    return data[index:index+24]['price']

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

def TrainLSTM(train_data, test_data, validate_data, features, layers=[20], save=False, num_epochs=50, b_size=100):

    # This is a wrapper function to train a (potentially multi-layered) Keras LSTM
    # Inputs are:
    # train, test and validation datasets (in the form of a pandas dataframe)
    # features (a list of strings representing columns in the train, test and validation sets) ((This also assumes that all features are scalable, so tread carefully!))
    # layers (an array of integers setting i) the number of total layers ( = len(layers)) and
    #                                      ii) the number of neurons in each layer
    #          examples: layers = [20,10,2] is a 3-hidden-layer LSTM with 20 neurons in its first layer, 10 neurons in its second layer and 2 neurons in its 3rd layer
    #                    layers = [20] is a 1-hidden-layer LSTM with 20 neurons in its only layer)
    # save (boolean which indicates whether output model should be saved to file or not. If True, name of the file will be set to number of neurons in each layer separated by underscore)


    mms = MinMaxScaler(feature_range=(0,1)) # Defining the scaling mechanism for our network

    # Making copies of training, validation and testing datasets
    val = validate_data.copy()
    val = val[features].values
    train = train_data.copy()
    train = train[features].values
    test = test_data.copy()
    test = test[features].values
    

    
    train_trans = mms.fit_transform(train) # Fitting our transformer on the training data
    test_trans = mms.transform(test) # Transforming the testing data here

    # Separating into inputs and outputs, then reshaping to appropriate shapes for network read-in
    train_X, train_Y = train_trans[:,1:], train_trans[:,0] 
    test_X, test_Y = test_trans[:,1:], test_trans[:,0]
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    train_Y = train_Y.reshape((train_Y.shape[0],))
    test_Y = test_Y.reshape((test_Y.shape[0],))

    # Uncomment for testing transformation
    # print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

    # multi-layered (Build and train model)
    model = Sequential()
    num_layers = len(layers)
    for index, layer in enumerate(layers):

        if index == 0:
            
            if index == (num_layers - 1):
                model.add(LSTM(layer,return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
            
            else:
                model.add(LSTM(layer,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
        
        elif index == (num_layers - 1):
            model.add(LSTM(layer,return_sequences=False))

        else:
            model.add(LSTM(layer, return_sequences=True))
    
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')

    history = model.fit(train_X,train_Y, epochs=num_epochs, 
                        batch_size=b_size, validation_data=(test_X,test_Y), 
                        verbose=0, shuffle=False)

    # Output final test set RMSE

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0],test_X.shape[2]))

    inv_yhat = np.concatenate((yhat,test_X),axis=1)
    inv_yhat = mms.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    test_Y = test_Y.reshape((len(test_Y),1))
    inv_y = np.concatenate((test_Y, test_X), axis=1)
    inv_y = mms.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    test_rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % test_rmse)

    # Look at validation dataset and calculate RMSE

    val_trans = mms.transform(val)
    fixed_X, fixed_Y = val_trans[:,1:], val_trans[:,0]
    fixed_X = fixed_X.reshape((fixed_X.shape[0],1,fixed_X.shape[1]))
    fixed_Y = fixed_Y.reshape((fixed_Y.shape[0],))

    val_yhat = model.predict(fixed_X)
    fixed_X = fixed_X.reshape((fixed_X.shape[0],fixed_X.shape[2]))
    inv_yhat = np.concatenate((val_yhat,fixed_X),axis=1)
    inv_yhat = mms.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    fixed_Y = fixed_Y.reshape((len(fixed_Y),1))
    inv_y = np.concatenate((fixed_Y, fixed_X), axis=1)
    inv_y = mms.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    validation_rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Validation RMSE: %.3f' % validation_rmse)

    if save:

        name = ''
        for layer in layers:
            
            if name:   
                name += str(layer)

            else:
                name += '_%s' % str(layer)
        
        model.save('%s.keras' % name)
    
    return model, history, test_rmse, validation_rmse 

def TrainLSTM24(train_data, test_data, validate_data, features, neurons=500, dense = 200, save=False, num_epochs=50, b_size=100):

    print('STARTING TRAINING MODEL')
    # This is a wrapper function to train a (potentially multi-layered) Keras LSTM to predict a full 24 hour window in one go!
    # Inputs are:
    # train, test and validation datasets (in the form of a pandas dataframe)
    # features (a list of strings representing columns in the train, test and validation sets) ((This also assumes that all features are scalable, so tread carefully!))
    # layers (an array of integers setting i) the number of total layers ( = len(layers)) and
    #                                      ii) the number of neurons in each layer
    #          examples: layers = [20,10,2] is a 3-hidden-layer LSTM with 20 neurons in its first layer, 10 neurons in its second layer and 2 neurons in its 3rd layer
    #                    layers = [20] is a 1-hidden-layer LSTM with 20 neurons in its only layer)
    # save (boolean which indicates whether output model should be saved to file or not. If True, name of the file will be set to number of neurons in each layer separated by underscore)


    mms = MinMaxScaler(feature_range=(0,1)) # Defining the scaling mechanism for our network

    # Making copies of training, validation and testing datasets
    val = validate_data.copy()
    val = val[features].values
    train = train_data.copy()
    train = train[features].values
    test = test_data.copy()
    test = test[features].values
    

    
    train_trans = mms.fit_transform(train) # Fitting our transformer on the training data
    test_trans = mms.transform(test) # Transforming the testing data here

    # Separating into inputs and outputs, then reshaping to appropriate shapes for network read-in
    train_X, train_Y = train_trans[:,:-24], train_trans[:,-24:] 
    test_X, test_Y = test_trans[:,:-24], test_trans[:,-24:]
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    train_Y = train_Y.reshape((train_Y.shape[0],train_Y.shape[1],1))
    test_Y = test_Y.reshape((test_Y.shape[0],test_Y.shape[1],1))

    # Uncomment for testing transformation
    # print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)

    # multi-layered (Build and train model)
    model = Sequential()
    model.add(LSTM(neurons,activation='relu',input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(RepeatVector(train_Y.shape[1]))
    model.add(LSTM(neurons,activation='relu',return_sequences=True))
    model.add(TimeDistributed(Dense(dense, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')

    model.fit(train_X, train_Y, epochs=num_epochs, batch_size=b_size, verbose=2)

    # Output final test set RMSE

    # yhat = model.predict(test_X)
    # test_X = test_X.reshape((test_X.shape[0],test_X.shape[2]))

    # inv_yhat = np.concatenate((yhat,test_X),axis=1)
    # inv_yhat = mms.inverse_transform(inv_yhat)
    # inv_yhat = inv_yhat[:,0]

    # test_Y = test_Y.reshape((len(test_Y),1))
    # inv_y = np.concatenate((test_Y, test_X), axis=1)
    # inv_y = mms.inverse_transform(inv_y)
    # inv_y = inv_y[:,0]

    # test_rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Test RMSE: %.3f' % test_rmse)

    # Look at validation dataset and calculate RMSE

    # val_trans = mms.transform(val)
    # fixed_X, fixed_Y = val_trans[:,1:], val_trans[:,0]
    # fixed_X = fixed_X.reshape((fixed_X.shape[0],1,fixed_X.shape[1]))
    # fixed_Y = fixed_Y.reshape((fixed_Y.shape[0],))

    # val_yhat = model.predict(fixed_X)
    # fixed_X = fixed_X.reshape((fixed_X.shape[0],fixed_X.shape[2]))
    # inv_yhat = np.concatenate((val_yhat,fixed_X),axis=1)
    # inv_yhat = mms.inverse_transform(inv_yhat)
    # inv_yhat = inv_yhat[:,0]

    # fixed_Y = fixed_Y.reshape((len(fixed_Y),1))
    # inv_y = np.concatenate((fixed_Y, fixed_X), axis=1)
    # inv_y = mms.inverse_transform(inv_y)
    # inv_y = inv_y[:,0]

    # validation_rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Validation RMSE: %.3f' % validation_rmse)

    # if save:

    #     name = ''
    #     for layer in layers:
            
    #         if name:   
    #             name += str(layer)

    #         else:
    #             name += '_%s' % str(layer)
        
    #     model.save('%s.keras' % name)
    
    return model


def main(loc='./'):

    ordered_train = pd.read_csv(loc+'ordered_train_set.csv',parse_dates=['time','date']).dropna()
    test_dataset  = pd.read_csv(loc+'ordered_test_set.csv',parse_dates=['time','date']).dropna()
    val_dataset   = pd.read_csv(loc+'ordered_seasonal_validation_set.csv',parse_dates=['time','date']).dropna()
    features = ['DA_price','hour','day_of_week','holiday','business_hour','temp',
                'dwpt','avg_load(h-24)','avg_DA_price(h-24)','RT_price(t-1D)',
                'DA_price(t-1D)', 'load(t-1D)','RT_price(t-2D)', 'DA_price(t-2D)',
                'load(t-2D)','RT_price(t-7D)', 'DA_price(t-7D)', 'load(t-7D)',
                'nat_gas_spot_price', 'monthly_avg_NY_natgas_price']
                    
    test = TrainLSTM(ordered_train,val_dataset,test_dataset,features,layers=[40,20,10],save=False)
    return test

if __name__ == '__main__':
    main()