"""
Created on Wed Sep 30 00:42:54 2020

@author: SadeghiTabas, Sadegh
===================================================
Applying the Kalman Filter for Missing Observations
===================================================

In this code we are going to apply `KalmanFilter` when some
measurements are missing.
While the Kalman Filter and Kalman Smoother are typically presented assuming a
measurement exists for every time step, this is not always the case in reality.
:class:`KalmanFilter` is implemented to recognize masked portions of numpy
arrays as missing measurements.
The figure drawn illustrates the trajectory of each dimension of the true
state, the estimated state using all measurements, and the estimated state
using every fifth measurement.
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import glob
import os
import pandas as pd
from numpy import ma
"""
# specify parameters
random_state = np.random.RandomState(0)
transition_matrix = [[1, 0.1], [0, 1]]
transition_offset = [-0.1, 0.1]
observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
observation_offset = [1.0, -1.0]
initial_state_mean = [5, -5]
n_timesteps = 50

"""
# sample from model
kf = KalmanFilter(
transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]]    
)

#kf = KalmanFilter(initial_state_mean=2.549, n_dim_obs=1)
files = glob.iglob(os.path.join(os.getcwd(),'Models/example', "*.txt"))
for file in files:
    file_path= file

col_names = ['Date', 'XXX', 'QObs','QObs2']
with open(file_path) as fp:
    df = pd.read_csv(fp, sep=';', header=None, names=col_names)
    dates = df.Date
    Qobs = df.QObs
C = Qobs._values
Qobs = np.array(C)
Z=np.array([Qobs,Qobs])
Qobs=Z.T
Qobs_origin = np.array(Qobs)
for t in range(len(Qobs)):
    if t > 5:
        if t < 10:
            Qobs[t] = np.ma.masked
            
"""
for t in range(len(Qobs)):
    if Qobs[t] == -999.000:
        Qobs[t] = np.ma.masked
"""
# estimate state with filtering and smoothing
#smoothed_states_all = kf.em(X).smooth(Qobs)
kf = kf.em(Qobs, n_iter=500)
(filtered_state_means, filtered_state_covariances) = kf.filter(Qobs)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(Qobs)


smoothed_states_missing = kf.smooth(Qobs)[0]
#predicted = np.reshape(smoothed_states_missing,(len(Qobs),))
predicted =smoothed_states_missing

"""
# draw estimates
pl.figure()
lines_true = pl.plot(dates,Qobs_origin,scalex=True, color='b')
#lines_smooth_all = pl.plot(smoothed_states_all, color='r')
lines_smooth_missing = pl.plot(dates,predicted,scalex=True, color='g')
pl.legend(
    (lines_true[0], lines_smooth_missing[0]),
    ('true', 'missing'),
    loc='lower right'
)
pl.show()
"""
start_date = pd.to_datetime(dates[0], format="%Y-%m-%d")
end_date = pd.to_datetime(dates[len(dates)-2], format="%Y-%m-%d")

start_date = start_date
end_date = end_date + pd.DateOffset(days=1)
date_range = pd.date_range(start_date, end_date)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(date_range, Qobs_origin, label="observation")
ax.plot(date_range, predicted, label="prediction")
ax.legend()
ax.set_title(f"Filling missing values from XXX to XXX")
ax.xaxis.set_tick_params(rotation=90)
ax.set_xlabel("Date")
_ = ax.set_ylabel("Discharge (mm/d)")
plt.show()
