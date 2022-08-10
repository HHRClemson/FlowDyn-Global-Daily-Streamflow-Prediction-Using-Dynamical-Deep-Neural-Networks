## CPSC8810 - Mining Massive Data Project

### Project Title: 
    A Web Platform for Dynamical Streamflow Prediction Using Machine Learning and Deep Learning Methods

### Contributors: 
    Sadegh Sadeghi Tabas
    Nushrat Humaira 
    
### Supervisors:
Dr. Vidya Samadi &
Dr. Nina Hubig

### Project Description: 
    In this research a number of data driven (machine learning and deep learning) and data mining methods
    including multi-layer perceptron (MLP), long short-term memory (LSTM) and a hybrid deep learning method 
    of convolutional neural network and LSTM have been implemented in a web designed platform to predict sequential
    flow rate values based on a set of collected runoff factors in a global scale (North America, South America and Africa).

### Dependencies:
    Tensorflow
    Keras
    Numpy
    Pandas
    Matplotlib
    Folium
    JS
    Jquery
    Leaflet
    Django
    ArcGIS Api

### Timeline:
|Num| Todo List | Deadline | Status |
| --- | --- | --- | --- |
|01| Download the Datasets from GRDC website | Sep 14 | Done! |
|02| Extract the Stations with less than 10% missing values | Sep 21 | Done! |
|03| Implement an Automatic ARIMA model to fill the missing values | Sep 28 | Done! |
|04| Submit the first report (Checkpoint 1) | Sep 30 | Done! |
|05| Implement a Machine Learning Method to Forecast Streamflow | Oct 28 | Done! |
|06| Submit Checkpoint 2 | Oct 31 | Done! |
|07| Submit Checkpoint 3 and final report| Nov 30 | Done! |
|08| Presentation | Dec 8| Working! |

## Repository info

### Datasets: 
    The input dataset retrieved from three sources as follows:
    1- GRDC website
    2- NCDS website
    3- CAMELS dataset

### Reports: 
Contains checkpoint reports
### Models:
    1. Two ipython notebooks fill the missing values using two different deep neural networks
    2. ARIMA.py replaces missing values with Autoregressive integrated moving average method
    3. GRDC_visualization.py performs analytics, given a world meterological union subregion, parse the grdc stations data and find geographically closest stations
       that are active the same time period
    4. folium.py visualizes station information and corresponding streamflow time series in a global map using folium package
    5. LSTM_Singlelayer, LSTM_CNN, MLP are three data driven models used in our project
    6. Streamflow prediction app with keras,django is an app to run inference on streamflow prediction model for one station
    7. encoder_decoder_lstm.py implements the seq2seq encoder-decoder LSTM network for future forecast and achived valid RMSE score of 116.46249
    8. Django Web Platform for Dynamical Streamflow Prediction with leaflet, jQuery (Please load test.html to start the webapp)




