import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
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

def TrainLSTM24(train_data, test_data, validate_data, features, neurons=500, dense = 200, save='', num_epochs=50, b_size=100):

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
    val_trans = mms.transform(val)


    # Separating into inputs and outputs, then reshaping to appropriate shapes for network read-in
    train_X, train_Y = train_trans[:,:-24], train_trans[:,-24:] 
    test_X, test_Y = test_trans[:,:-24], test_trans[:,-24:]
    val_X, val_Y = val_trans[:,:-24], val_trans[:,-24:]
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0],1,val_X.shape[1]))
    train_Y = train_Y.reshape((train_Y.shape[0],train_Y.shape[1],1))
    test_Y = test_Y.reshape((test_Y.shape[0],test_Y.shape[1],1))
    val_Y = val_Y.reshape((val_Y.shape[0],val_Y.shape[1],1))

    # Uncomment for testing transformation
    # print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)
    callback = EarlyStopping(monitor='loss',patience=8)
    # multi-layered (Build and train model)
    model = Sequential()
    model.add(LSTM(neurons,activation='relu',input_shape=(train_X.shape[1],train_X.shape[2])))
    model.add(RepeatVector(train_Y.shape[1]))
    model.add(LSTM(neurons,activation='relu',return_sequences=True))
    model.add(TimeDistributed(Dense(dense, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    # model.add(Dense(dense, activation='relu'))
    # model.add(Dense(train_Y.shape[1]))
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt)

    model.fit(train_X, train_Y, epochs=num_epochs, batch_size=b_size, verbose=2, callbacks=[callback])

    val_yhat, val_y, val_rmse = get_predictions_model(val_X, val_Y, model, mms)
    test_yhat, test_y, test_rmse = get_predictions_model(test_X, test_Y, model, mms)
    print('Test RMSE %.3f' % val_rmse)
    print('Validation RMSE %.3f' % test_rmse)


    if save:
        
        model.save('%s.keras' % save)
    
    return model, [val_yhat, val_y], [test_yhat, test_y]

def get_rmse_sample(actual,predicted):

    return np.sqrt(mean_squared_error(actual,predicted))

def get_predictions_model(X,act_Y,model,scaler):

    # X and Y are already in the correct shape (as they've been dealt with inside the fitting function)

    yhat = model.predict(X, verbose=1)
    all_yhat = np.concatenate((X.reshape(X.shape[0],X.shape[2]),yhat.reshape((yhat.shape[0],yhat.shape[1]))), axis=1)
    yhat_inv = scaler.inverse_transform(all_yhat)
    yhat_y = yhat_inv[:,-24:]

    all_y = np.concatenate((X.reshape(X.shape[0],X.shape[2]),act_Y.reshape((act_Y.shape[0],act_Y.shape[1]))), axis=1)
    act_inv = scaler.inverse_transform(all_y)
    y = act_inv[:,-24:]

    return yhat_y, y, get_rmse_sample(y,yhat_y)

def prediction_routine(train_file,test_file,model_file,features):

    train_dat = load_data(train_file,features)
    mod = load_model(model_file)
    scaler = fit_scaler(train_dat)
    test_dat = load_data(test_file,features)
    test_x, test_y = transform_reshape(test_dat,scaler)
    yhat, y, rmse = get_predictions_model(test_x,test_y,mod,scaler)

    return yhat, y, rmse, scaler


def transform_reshape(data,scaler):

    trans = scaler.transform(data.copy())
    X,Y = trans[:,:-24], trans[:,-24:]
    X = X.reshape((X.shape[0],1,X.shape[1]))
    Y = Y.reshape((Y.shape[0],Y.shape[1],1))
    return X, Y

def load_data(filename,features):

    temp = pd.read_csv(filename,parse_dates=['time','date']).dropna()
    return temp[features].values

def fit_scaler(data):

    mms = MinMaxScaler(feature_range=(0,1))
    curr = data.copy()
    trans = mms.fit_transform(curr)
    return mms

def save_plots(model,filename,features,scaler,nplots=5):

    dset = pd.read_csv(filename,parse_dates=['time','date']).dropna()
    dset_clean = load_data(filename,features)
    for ipl in range(nplots):
        curr_day = dset.sample(1).index[0] # get index of the chosen day
        date_range = dset[curr_day-24:curr_day]
        ext_date_range = dset[curr_day:curr_day+24]
        wholedate = dset[curr_day-24:curr_day+24]
        #print(dset_clean.shape)
        dat = dset_clean[curr_day].reshape((1,dset_clean.shape[1]))
        #print(dat.shape)
        dat_x, dat_y = transform_reshape(dat,scaler)
        pred, act, rmse = get_predictions_model(dat_x,dat_y,model,scaler)
        base_rmse = np.sqrt(mean_squared_error(act[0],ext_date_range['DA_price(t-1D)']))
        #print(rmse)
        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(wholedate.time,wholedate.DA_price,'b',label='Actual Price')
        ax.plot(ext_date_range.time, pred[0], 'g-^', label='Prediction')
        #ax.plot(ext_date_range.time, ext_date_range.DA_price, 'b',label='Actual')
        ax.plot(ext_date_range.time, ext_date_range['DA_price(t-1D)'].values,'r-o',label='Baseline')
        plt.text(0.5,0.9,r'Model RMSE = %.3f' % rmse, transform=ax.transAxes,fontsize=25)
        plt.text(0.15,0.9,r'Baseline RMSE = %.3f' % base_rmse, transform=ax.transAxes,fontsize=25)
        ax.set_xlabel('Date & Time', fontsize=25)
        ax.set_ylabel('Day-Ahead Price', fontsize=25)
        ax.tick_params(axis='x',labelsize=20)
        ax.tick_params(axis='y',labelsize=20)
        
        plt.legend(loc='lower left',fontsize=25)
        plt.savefig(r'%s.png' % str(ipl), format='png')
        plt.close()



def get_features():

    return ['DA_price',
       'RT_price', 'load', 'temp', 'dwpt', 'nat_gas_spot_price',
       'monthly_avg_NY_natgas_price',
       'load(h-1)', 'load(h-2)',
       'load(h-19)', 'load(h-20)',
       'load(h-21)','load(h-22)',
       'load(h-23)','load(h-24)',
       'load(h-25)', 'load(h-26)',
       'load(h-49)', 'load(h-168)', 'hour', 'day',
       'weekday', 'month', 'day_of_week', 'holiday', 'business_hour',
       'season', 'avg_RT_price_prev_day', 'avg_actual_load_prev_day',
       'RT_price(t-1D)', 'load(t-1D)',
       'RT_price(t-2D)', 'load(t-2D)', 'DA_price(t-3D)', 'RT_price(t-3D)',
       'load(t-3D)', 'DA_price(t-4D)', 'RT_price(t-4D)', 'load(t-4D)',
       'DA_price(t-5D)', 'RT_price(t-5D)', 'load(t-5D)', 'DA_price(t-6D)',
       'RT_price(t-6D)', 'load(t-6D)', 'DA_price(t-7D)', 'RT_price(t-7D)',
       'load(t-7D)',
       'DA_price(t-1h)', 'DA_price(t-2h)', 'DA_price(t-3h)',
       'DA_price(t-4h)', 'DA_price(t-5h)', 'DA_price(t-6h)',
       'DA_price(t-7h)', 'DA_price(t-8h)', 'DA_price(t-9h)',
       'DA_price(t-10h)', 'DA_price(t-11h)', 'DA_price(t-12h)',
       'DA_price(t-13h)', 'DA_price(t-14h)', 'DA_price(t-15h)',
       'DA_price(t-16h)', 'DA_price(t-17h)', 'DA_price(t-18h)',
       'DA_price(t-19h)', 'DA_price(t-20h)', 'DA_price(t-21h)',
       'DA_price(t-22h)', 'DA_price(t-23h)', 'DA_price(t-24h)',
       'DA_price(t-25h)', 'DA_price(t-26h)', 'DA_price(t-27h)',
       'DA_price(t-28h)', 'DA_price(t-29h)', 'DA_price(t-30h)',
       'DA_price(t-31h)', 'DA_price(t-32h)', 'DA_price(t-33h)',
       'DA_price(t-34h)', 'DA_price(t-35h)', 'DA_price(t-36h)',
       'DA_price(t-37h)', 'DA_price(t-38h)', 'DA_price(t-39h)',
       'DA_price(t-40h)', 'DA_price(t-41h)', 'DA_price(t-42h)',
       'DA_price(t-43h)', 'DA_price(t-44h)', 'DA_price(t-45h)',
       'DA_price(t-46h)', 'DA_price(t-47h)', 'DA_price(t-48h)', 
       'DA_price(t+1h)', 'DA_price(t+2h)', 'DA_price(t+3h)',
       'DA_price(t+4h)', 'DA_price(t+5h)', 'DA_price(t+6h)',
       'DA_price(t+7h)', 'DA_price(t+8h)', 'DA_price(t+9h)',
       'DA_price(t+10h)', 'DA_price(t+11h)', 'DA_price(t+12h)',
       'DA_price(t+13h)', 'DA_price(t+14h)', 'DA_price(t+15h)',
       'DA_price(t+16h)', 'DA_price(t+17h)', 'DA_price(t+18h)',
       'DA_price(t+19h)', 'DA_price(t+20h)', 'DA_price(t+21h)',
       'DA_price(t+22h)', 'DA_price(t+23h)', 'DA_price(t+24h)']


    

def main():

    ordered_train = pd.read_csv('../data_processing/final_data/ordered_train_set.csv',parse_dates=['time','date']).dropna()
    test_dataset  = pd.read_csv('../data_processing/final_data/ordered_test_set.csv',parse_dates=['time','date']).dropna()
    val_dataset   = pd.read_csv('../data_processing/final_data/ordered_seasonal_validation_set.csv',parse_dates=['time','date']).dropna()
    # features = ['DA_price',
    #    'RT_price', 'load', 'temp', 'dwpt', 'nat_gas_spot_price',
    #    'monthly_avg_NY_natgas_price',
    #    'load(h-1)', 'load(h-2)',
    #    'load(h-19)', 'load(h-20)',
    #    'load(h-21)','load(h-22)',
    #    'load(h-23)','load(h-24)',
    #    'load(h-25)', 'load(h-26)',
    #    'load(h-49)', 'load(h-168)', 'hour', 'day',
    #    'weekday', 'month', 'day_of_week', 'holiday', 'business_hour',
    #    'season', 'avg_RT_price_prev_day', 'avg_actual_load_prev_day',
    #  'load(t-1D)',
    #  'load(t-2D)', 'DA_price(t-3D)', 
    #    'load(t-3D)', 'DA_price(t-4D)', 'load(t-4D)',
    #    'DA_price(t-5D)', 'load(t-5D)', 'DA_price(t-6D)',
    #    'load(t-6D)', 'DA_price(t-7D)',
    #    'load(t-7D)',
    #    'DA_price(t-1h)', 'DA_price(t-2h)', 'DA_price(t-3h)',
    #    'DA_price(t-4h)', 'DA_price(t-5h)', 'DA_price(t-6h)',
    #    'DA_price(t-7h)', 'DA_price(t-8h)', 'DA_price(t-9h)',
    #    'DA_price(t-10h)',  'DA_price(t-12h)',
    #     'DA_price(t-14h)', 
    #    'DA_price(t-16h)',  'DA_price(t-18h)',
    #     'DA_price(t-20h)', 
    #    'DA_price(t-22h)',  'DA_price(t-24h)',
    #     'DA_price(t-26h)', 
    #    'DA_price(t-28h)',  'DA_price(t-30h)',
    #  'DA_price(t-32h)',
    #    'DA_price(t-34h)', 'DA_price(t-36h)',
    #  'DA_price(t-38h)',
    #    'DA_price(t-40h)', 'DA_price(t-42h)',
    #  'DA_price(t-44h)',
    #    'DA_price(t-46h)', 'DA_price(t-48h)', 
    #    'DA_price(t+1h)', 'DA_price(t+2h)', 'DA_price(t+3h)',
    #    'DA_price(t+4h)', 'DA_price(t+5h)', 'DA_price(t+6h)',
    #    'DA_price(t+7h)', 'DA_price(t+8h)', 'DA_price(t+9h)',
    #    'DA_price(t+10h)', 'DA_price(t+11h)', 'DA_price(t+12h)',
    #    'DA_price(t+13h)', 'DA_price(t+14h)', 'DA_price(t+15h)',
    #    'DA_price(t+16h)', 'DA_price(t+17h)', 'DA_price(t+18h)',
    #    'DA_price(t+19h)', 'DA_price(t+20h)', 'DA_price(t+21h)',
    #    'DA_price(t+22h)', 'DA_price(t+23h)', 'DA_price(t+24h)']

    features = ['DA_price',
       'RT_price', 'load', 'temp', 'dwpt', 'nat_gas_spot_price',
       'monthly_avg_NY_natgas_price',
       'load(h-1)', 'load(h-2)',
       'load(h-19)', 'load(h-20)',
       'load(h-21)','load(h-22)',
       'load(h-23)','load(h-24)',
       'load(h-25)', 'load(h-26)',
       'load(h-49)', 'load(h-168)', 'hour', 'day',
       'weekday', 'month', 'day_of_week', 'holiday', 'business_hour',
       'season', 'avg_RT_price_prev_day', 'avg_actual_load_prev_day',
       'RT_price(t-1D)', 'load(t-1D)',
       'RT_price(t-2D)', 'load(t-2D)', 'DA_price(t-3D)', 'RT_price(t-3D)',
       'load(t-3D)', 'DA_price(t-4D)', 'RT_price(t-4D)', 'load(t-4D)',
       'DA_price(t-5D)', 'RT_price(t-5D)', 'load(t-5D)', 'DA_price(t-6D)',
       'RT_price(t-6D)', 'load(t-6D)', 'DA_price(t-7D)', 'RT_price(t-7D)',
       'load(t-7D)',
       'DA_price(t-1h)', 'DA_price(t-2h)', 'DA_price(t-3h)',
       'DA_price(t-4h)', 'DA_price(t-5h)', 'DA_price(t-6h)',
       'DA_price(t-7h)', 'DA_price(t-8h)', 'DA_price(t-9h)',
       'DA_price(t-10h)', 'DA_price(t-11h)', 'DA_price(t-12h)',
       'DA_price(t-13h)', 'DA_price(t-14h)', 'DA_price(t-15h)',
       'DA_price(t-16h)', 'DA_price(t-17h)', 'DA_price(t-18h)',
       'DA_price(t-19h)', 'DA_price(t-20h)', 'DA_price(t-21h)',
       'DA_price(t-22h)', 'DA_price(t-23h)', 'DA_price(t-24h)',
       'DA_price(t-25h)', 'DA_price(t-26h)', 'DA_price(t-27h)',
       'DA_price(t-28h)', 'DA_price(t-29h)', 'DA_price(t-30h)',
       'DA_price(t-31h)', 'DA_price(t-32h)', 'DA_price(t-33h)',
       'DA_price(t-34h)', 'DA_price(t-35h)', 'DA_price(t-36h)',
       'DA_price(t-37h)', 'DA_price(t-38h)', 'DA_price(t-39h)',
       'DA_price(t-40h)', 'DA_price(t-41h)', 'DA_price(t-42h)',
       'DA_price(t-43h)', 'DA_price(t-44h)', 'DA_price(t-45h)',
       'DA_price(t-46h)', 'DA_price(t-47h)', 'DA_price(t-48h)', 
       'DA_price(t+1h)', 'DA_price(t+2h)', 'DA_price(t+3h)',
       'DA_price(t+4h)', 'DA_price(t+5h)', 'DA_price(t+6h)',
       'DA_price(t+7h)', 'DA_price(t+8h)', 'DA_price(t+9h)',
       'DA_price(t+10h)', 'DA_price(t+11h)', 'DA_price(t+12h)',
       'DA_price(t+13h)', 'DA_price(t+14h)', 'DA_price(t+15h)',
       'DA_price(t+16h)', 'DA_price(t+17h)', 'DA_price(t+18h)',
       'DA_price(t+19h)', 'DA_price(t+20h)', 'DA_price(t+21h)',
       'DA_price(t+22h)', 'DA_price(t+23h)', 'DA_price(t+24h)']
                    
    test = TrainLSTM24(ordered_train,test_dataset,val_dataset,features,num_epochs=200,neurons=300,dense=200,b_size=72,save=sys.argv[1])
    

if __name__ == '__main__':
    main()