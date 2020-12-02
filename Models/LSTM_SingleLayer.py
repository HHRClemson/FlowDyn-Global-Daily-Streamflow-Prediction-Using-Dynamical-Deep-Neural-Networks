# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:20:55 2020

@author: SadeghiTabas
@author: Nushrat
"""


import numpy
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas
import math
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path

def dropconnect_regularizer(weight_matrix):
    return tf.nn.dropout(weight_matrix,rate=0.2)
p = Path('/home/nhumair/CPSC8810-Mining-Massive-Data/Models/US_Data/Untitled_Folder')
for fn in p.glob("*.csv"):
    print(os.path.basename(fn))
    #print(os.path.dirname(fn))
    dir_name = os.path.dirname(fn)
    savefilename = os.path.basename(fn).split(".")[0]
    #print(savefilename)
    # In[-]: Importing the dataset
    datasets = pd.read_csv(fn)
    #print(datasets.columns)
    datasets = datasets.rename(columns={'Unnamed: 3': 'Value_1'})
    datasets.loc[0,'Value_1']= datasets.loc[0,' Value']
    for i in range(1, len(datasets)):
        datasets.loc[i, 'Value_1'] = datasets.loc[i-1, ' Value']
    #print(datasets.head(20))
    #datasets.drop(datasets.columns[4], axis=1,inplace=True) # had to this for capetown file
    #datasets.drop(datasets.columns[6], axis=1,inplace=True)
    #print(datasets.head(20))
    datasets = datasets.dropna()
    #print(len(datasets))
    num_examples = len(datasets)
    num_train = int(0.7 * num_examples)
    train_examples_x = datasets.iloc[:num_train,2:4].values.astype(float)
    train_examples_y = datasets.iloc[:num_train,4:5].values.astype(float)
    val_examples_x = datasets.iloc[num_train:,2:4].values.astype(float)
    val_examples_y = datasets.iloc[num_train:,4:5].values.astype(float)
    dates_array = datasets.iloc[num_train:,1:2].values
    #dates_array = dates_array.reshape(-1,1)
    #print(dates_array)
    
    dates_test = datasets.iloc[num_train:,1].values.tolist()
    print(len(dates_test))
    dates_train = datasets.iloc[:num_train,1].values.tolist()

    #x_train = datasets.iloc[7305:10958, 2:8].values.astype(float)   # Prc, Srad, Tmax, Tmin, VP, runoff (t-1)
    
    #train, test = train_test_split(datasets, test_size=0.3)
    #x_train = train.iloc[:, 2:4].values.astype(float)   # Prc,runoff (t-1)
    #y_train = train.iloc[:, 4:5].values.astype(float)  # runoff (t)
    #x_test= test.iloc[:, 2:4].values.astype(float)
    #y_test= test.iloc[:, 4:5].values.astype(float)

    # In[-]: Feature Scaling

    from sklearn.preprocessing import StandardScaler
    sc_x_train = StandardScaler()
    sc_y_train = StandardScaler()
    x_train = sc_x_train.fit_transform(train_examples_x)
    y_train = sc_y_train.fit_transform(train_examples_y).ravel()
    x_test=sc_x_train.transform(val_examples_x)
    observation_train = train_examples_y
    observation_test = val_examples_y

    # In[]: LSTM
    # fix random seed for reproducibility
    numpy.random.seed(7)


    # create and fit the LSTM network
    x_train = array(x_train).reshape(numpy.shape(x_train)[0], 1, 2)
    model = Sequential()
    #model.add(LSTM(30, activation='relu', input_shape=(1, 2)))
    model.add(LSTM(30, recurrent_regularizer=dropconnect_regularizer,activation='relu',return_sequences=True,input_shape=(1, 2)))
    model.add(LSTM(30,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=2)

    #model.save('model.hdf5')
    # make predictions
    x_test = array(x_test).reshape(numpy.shape(x_test)[0], 1, 2)
    LSTM_train_predict = model.predict(x_train)
    LSTM_test_predict = model.predict(x_test)

    # invert predictions
    LSTM_train_Sim = sc_y_train.inverse_transform(LSTM_train_predict)
    LSTM_test_Sim = sc_y_train.inverse_transform(LSTM_test_predict)

    #print(observation[0])
    #print("\n")
    #print(LSTM_test_Sim[0])
    #print(observation.shape)
    #print(LSTM_test_Sim.shape)
   
    combined_array_1 = numpy.concatenate([observation_test,LSTM_test_Sim],axis=1)
    #print(combined_array)
    combined_array_2 = numpy.concatenate([observation_train,LSTM_train_Sim],axis=1)
    total_path_dates=dir_name+"/"+savefilename+"_dates_lstm.csv"
    #numpy.savetxt(total_path_dates, dates_array, fmt='%s')
    total_path_test=dir_name+"/"+savefilename+"_test_result_lstm.csv"
    numpy.savetxt(total_path_test, combined_array_1, delimiter=",")
    total_path_train = dir_name+"/"+savefilename+"_train_result_lstm.csv"
    #numpy.savetxt(total_path_train,combined_array_2,delimiter=",")
    # In[]: Result

    print(f"LSTM: {r2_score(observation_test, LSTM_test_Sim):.2f}" )
    
    
    
    
        
    date_range_test=[]
    for date in dates_test:
        d_parsed = pd.to_datetime(date, format="%m/%d/%Y")
        date_range_test.append(d_parsed)  # add transformed date to new list
    date_range_train = []
    for date in dates_train:
        d_parsed = pd.to_datetime(date,format="%m/%d/%Y")
        date_range_train.append(d_parsed)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(date_range_test, observation_test, label="observation")
    ax.plot(date_range_test, LSTM_test_Sim, label="simulation")
    ax.legend()
    ax.set_title(f"Basin: {savefilename} NSE={r2_score(observation_test, LSTM_test_Sim):.2f}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Date")
    _ = ax.set_ylabel("Discharge (mm/d)")
    plt.savefig(dir_name+"/"+savefilename+"_lstm_test.png")
    fig,ax = plt.subplots(figsize=(12,4))
    ax.plot(date_range_train,observation_train,label="observation")
    ax.plot(date_range_train,LSTM_train_Sim, label="simulation")
    ax.legend()
    ax.set_title(f"Basin: {savefilename} NSE={r2_score(observation_train,LSTM_train_Sim):.2f}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Date")
    _ = ax.set_ylabel("Dischard (mm/d)")
    plt.savefig(dir_name+"/"+savefilename+"_lstm_train.png")
    
    print("done!")
    
print("finished\n")
