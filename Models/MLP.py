# -*- coding: utf-8 -*-
"""

@author: Sadegh
@author: Nushrat
"""
# In[-]: import libraries
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from numpy import savetxt
#from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import pandas
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os
from sklearn.model_selection import train_test_split


p = Path('/home/nhumair/CPSC8810-Mining-Massive-Data/Models/SA/SA_new')
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
    
    datasets = datasets.dropna()
    #print(len(datasets))
    num_examples = len(datasets)
    num_train = int(0.7 * num_examples)
    train_examples_x = datasets.iloc[:num_train,2:4].values.astype(float)
    train_examples_y = datasets.iloc[:num_train,4:5].values.astype(float)
    val_examples_x = datasets.iloc[num_train:,2:4].values.astype(float)
    val_examples_y = datasets.iloc[num_train:,4:5].values.astype(float)
    
    dates_array = datasets.iloc[num_train:,1:2].values
    
    dates_test = datasets.iloc[num_train:,1].values.tolist()
    print(len(dates_test))
    dates_train = datasets.iloc[:num_train,1].values.tolist()
    
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
    np.random.seed(7)


    # In[1]: MLP

    MLP = MLPRegressor(activation='relu', hidden_layer_sizes=(20), learning_rate='adaptive',
              max_iter=10000, random_state=42, solver='sgd')
    MLP.fit(x_train, y_train)

    MLP_Pred_test = MLP.predict(x_test)
    MLP_Pred_train = MLP.predict(x_train)
    MLP_Sim_test = sc_y_train.inverse_transform(MLP_Pred_test)
    MLP_Sim_train = sc_y_train.inverse_transform(MLP_Pred_train)
    print('Nash–Sutcliffe model efficiency coefficient:')
    print(f"MLP: {r2_score(observation_test, MLP_Sim_test):.2f}" )
    

    #combined_array = numpy.concatenate([observation,MLP_Sim],axis=1)
    #print(combined_array)
    total_path=dir_name+"/"+savefilename+"_obser_mlp_result.csv"
    #np.savetxt(total_path, observation_test, delimiter=",")
    total_path = dir_name + "/" + savefilename + "_simul_mlp_result.csv"
    #np.savetxt(total_path, MLP_Sim_test, delimiter=",")


    
    

    #combined_array = np.concatenate([observation,MLP_Sim],axis=1)
    #print(combined_array)
    #total_path=dir_name+"/"+savefilename+"_mlp_sim.csv"
    #np.savetxt(total_path, MLP_Sim, delimiter=",")
    
    
    
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
    ax.plot(date_range_test, MLP_Sim_test, label="simulation")
    ax.legend()
    ax.set_title(f"Basin: {savefilename} NSE={r2_score(observation_test, MLP_Sim_test):.2f}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Date")
    _ = ax.set_ylabel("Discharge (mm/d)")
    plt.savefig(dir_name+"/"+savefilename+"_mlp_test.png")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(date_range_train, observation_train, label="observation")
    ax.plot(date_range_train, MLP_Sim_train, label="simulation")
    ax.legend()
    ax.set_title(f"Basin: {savefilename} NSE={r2_score(observation_train, MLP_Sim_train):.2f}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Date")
    _ = ax.set_ylabel("Discharge (mm/d)")
    plt.savefig(dir_name+"/"+savefilename+"_mlp_train.png")
    
    print("done!")
    
print("finished\n")    
    

