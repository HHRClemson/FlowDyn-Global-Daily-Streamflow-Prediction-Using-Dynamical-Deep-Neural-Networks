"""

@author: Nushrat
"""



import numpy as np

import tensorflow as tf

import random as rn

import numpy as np



import os

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)



from keras import backend as K



tf.set_random_seed(1234)
#tf.random.set_seed(1234)


sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#sess = tf.Session(graph=tf.get_default_graph())

K.set_session(sess)

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.layers import Input, Dense,LSTM,RepeatVector,GRU,Dropout,Reshape

from keras.layers import*

from keras.models import Model

from keras.models import Sequential

from deap import base, creator, tools, algorithms

from pandas import DataFrame

from pandas import Series

from pandas import concat

from pandas import read_csv

from pandas import datetime

import matplotlib.pyplot as plt

from math import sqrt
from tensorflow.contrib.rnn import *

import random

import warnings

warnings.simplefilter("ignore", DeprecationWarning)


def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset[i] - dataset[i - interval]

        diff.append(value)

    return Series(diff)

# invert differenced value

def inverse_difference(history, yhat, interval=1):

    return yhat + history[:,0][-interval]

# scale train and test data to [-1, 1]

def scale(train, test):

    # fit scaler

    scaler = MinMaxScaler(feature_range=(-1, 1))

    scaler = scaler.fit(train)

    # transform train

    train = train.reshape(train.shape[0], train.shape[1])

    train_scaled = scaler.transform(train)

    # transform test

    test = test.reshape(test.shape[0], test.shape[1])

    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled



# inverse scaling for a forecasted value

def invert_scale(scaler, X, value):

    new_row = []

    for x in X:

        new_row = new_row+[i for i in x]

            

    new_row.append(value) 

    new_row_2 = np.array(new_row)

    new_row_2 = new_row_2.reshape((1,new_row_2.shape[0]))

    inverted = scaler.inverse_transform(new_row_2)

    return inverted[0, -1]
def create_dataset(dataset,features, look_back=1):

    dataset = np.insert(dataset,[0]*look_back,0)    

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back):

        a = dataset[i:(i+look_back)]

        dataX.append(a)

        dataY.append(dataset[i + look_back])

    dataY= np.array(dataY)       

    dataY = np.reshape(dataY,(dataY.shape[0],features))

    dataset = np.concatenate((dataX,dataY),axis=1)  

    return dataset

# make a one-step forecast

def forecast_lstm(model, batch_size, X):

    X = X.reshape(batch_size, X.shape[0], X.shape[1])

    yhat = model.predict(X, batch_size=batch_size)

    return yhat[0,0]


# convert series to supervised learning

def series_to_supervised(data,features, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    x = np.zeros(features,dtype=np.int8)

    for i in range(n_in):

        data = np.insert(data,x,0)

    data = data.reshape(int(data.shape[0]/features),features) 

    df = DataFrame(data)
    #print(df.shape)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):
        
        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    #print("here!")
    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    #print("here2!")
    # put it all together

    agg = concat(cols, axis=1)

    agg.columns = names
    print(agg.columns)
    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)
    #print(agg.shape)
    return agg
def read_data(): 


    window_size = 0
    #features = 8
    features = 7





    series = read_csv('dataset - Copy.csv', header=0, index_col=0)	

    raw_values = series.values
    print(raw_values)



    # integer encode wind direction

    #encoder = LabelEncoder()

    #raw_values[:,4] = encoder.fit_transform(raw_values[:,4])



    # transform data to be stationary
    diff = difference(raw_values, 1)





    dataset = diff.values

    dataset = create_dataset(dataset,features,window_size)




    return dataset,raw_values

data,raw_values = read_data()
print(np.shape(data))
#print(data.dtype)
print(type(data[1][6]))

def build_model():    





    features = 7


    # frame as supervised learning


    reframed = series_to_supervised(data,features, 5, 1)
    #print(reframed.shape)


    drop = [i for  i in  range(5*features+1,((5+1)*features))]

    reframed.drop(reframed.columns[drop], axis=1, inplace=True)
    print("after drop {}".format(reframed.shape))
    reframed = reframed.values





    # split into train and test sets

    #train_size = 365*24*4
    #print(int(len(reframed)*0.7))
    train_size = int(len(reframed)*0.7)

    train, test = reframed[0:train_size], reframed[train_size:]
    print(np.shape(train))
    print(np.shape(test))




    # transform the scale of the data

    scaler, train_scaled, test_scaled = scale(train, test)



    # divide train into train and valid

    #train2_size = int(365*24*3.5)
    train2_size = int(len(train_scaled)*0.8)

    train_scaled2, valid = train_scaled[0:train2_size], train_scaled[train2_size:]





    # split into input and outputs

    x_train,y_train = train_scaled2[:,0:-1],train_scaled2[:,-1]

    x_valid,y_valid = valid[:,0:-1],valid[:,-1]

    x_test,y_test = test_scaled[:,0:-1],test_scaled[:,-1]



    # reshape input to be 3D [samples, timesteps, features]

    x_train = x_train.reshape(x_train.shape[0],5,features)

    x_valid = x_valid.reshape(x_valid.shape[0],5,features)

    x_test = x_test.reshape(x_test.shape[0],5,features)

    print(x_train.shape, y_train.shape,x_valid.shape,y_valid.shape, x_test.shape, y_test.shape)





    print('\nstart pretraining')

    print('===============')



    # train AE

    timesteps = x_train.shape[1]

    input_dim = x_train.shape[2]
    #print((params['batch_size'],timesteps, input_dim))
    print((73,timesteps, input_dim))

    AE = Sequential()

 
    AE.add(LSTM(20,input_shape=(timesteps, input_dim),stateful = False))

    AE.add(RepeatVector(timesteps))

 
    AE.add(LSTM(input_dim,stateful = False,return_sequences = True))


    print(x_train.shape)
    AE.compile(loss='mean_squared_error', optimizer='Adam',metrics=[tf.keras.metrics.MeanSquaredError()])


    #omitted  batch_size= 32,
    #AE.fit(x_train, x_train, epochs=params['epochs_pre'],shuffle=True,verbose = 0)
    AE.fit(x_train, x_train, epochs=100,shuffle=True,verbose = 0)





    trained_encoder = AE.layers[0]

    weights = AE.layers[0].get_weights()







    # Fine-turning

    print('\nFine-turning')

    print('============')



    #build finetuning model

    model = Sequential()

    model.add(trained_encoder)

    model.layers[-1].set_weights(weights)

    
    model.add(Dropout(0.2))

    model.add(Dense(1))



    model.compile(loss='mean_squared_error', optimizer='Adam')



    
    model.fit(x_train, y_train, epochs= 100, verbose = 0,shuffle=True) # 390 epochs for fine tune?





    # redefine the model in order to test with one sample at a time (batch_size = 1)

    new_model = Sequential()


    new_model.add(LSTM(20,batch_input_shape=( 1,timesteps, input_dim),stateful = False))

    
    new_model.add(Dropout(0.2))

    new_model.add(Dense(1))



    # copy weights

    old_weights = model.get_weights()

    new_model.set_weights(old_weights)





    # forecast the valid data

    print('Forecasting valid Data')

    predictions_valid = list()

    for i in range(len(valid)):

        # make one-step forecast

        X = x_valid[i]

        y= y_valid[i]



        yhat = forecast_lstm(new_model, 1, X)

        # invert scaling

        yhat = invert_scale(scaler, X, yhat)

        # invert differencing

        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+len(valid)+1-i)

        # store forecast

        predictions_valid.append(yhat)

        expected = raw_values[:,0][len(train_scaled2) + i + 1]
        print(type(expected))
        print('==============================')
        print(expected.shape)

        #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))



    # report performance using RMSE

    rmse_valid = sqrt(mean_squared_error(raw_values[:,0][len(train_scaled2)+1:len(valid)+len(train_scaled2)+1], predictions_valid))

    print('valid RMSE: %.5f' % rmse_valid)

    print('==============================')
    
    

    return {'loss': rmse_valid, 'status': STATUS_OK}




build_model()