"""
@author: Nushrat

visualization stations using folium package
"""

import folium
import os
from folium import features
import numpy as np
import pandas as pd
import vincent
from sklearn.preprocessing import MinMaxScaler
from folium import Map, CircleMarker, Vega, Popup,IFrame
import json
import base64


df = pd.read_csv('allregion_stations.csv', sep=',')
df = df.dropna()
df['lat'] = df.loc[:,['lat']].astype('float64')
df['long'] = df.loc[:,['long']].astype('float64')

lons = df['long'][0:1000].values
lats = df['lat'][0:1000].values

data = {
   'type': 'FeatureCollection',
   'features': [
      {
         'type': 'Feature',
         'geometry': {
            'type': 'MultiPoint',
            'coordinates': [[lon, lat] for (lat, lon) in zip(lats, lons)],
         },
         'properties': {'prop0': 'value0'}
      },
   ],
}

latitude = 37.0902
longitude = -95.7129
us_map = folium.Map(location=[latitude, longitude], zoom_start=2)
us_map.add_child(features.GeoJson(data))
us_map.save('Features.html')
##############################
us_map_2 = folium.Map(location=[latitude, longitude], zoom_start=2)

df = pd.read_csv('allregion_stations_2.csv', sep=',')
df = df.dropna()
df['lta_discharge'] = df['lta_discharge'].replace(['n.a'],float(0.0))
#print(type(df_data['lat']))
df['lat'] = df.loc[:,['lat']].astype('float64')
df['long'] = df.loc[:,['long']].astype('float64')
df['area'] = df.loc[:,['area']].astype('float64')
df['lta_discharge'] = df.loc[:,['lta_discharge']].astype('float64')
df['r_volume_yr'] = df.loc[:,['r_volume_yr']].astype('float64')
df['grdc_no']= df.loc[:,['grdc_no']]
############ncdc stations us##############
df_2 = pd.read_csv('newstations.csv', sep=',')
#df_2 = df_2.dropna()
#df['lta_discharge'] = df['lta_discharge'].replace(['n.a'],float(0.0))
#print(type(df_data['lat']))
df_2['LATITUDE'] = df_2.loc[:,['LATITUDE']].astype('float64')
df_2['LONGITUDE'] = df_2.loc[:,['LONGITUDE']].astype('float64')
df_2['STATION_ID']= df_2.loc[:,['STATION_ID']]
df_2['ELEVATION_(M)']= df_2.loc[:,['ELEVATION_(M)']]

station_basins = f'/home/nhumair/CPSC8810-Mining-Massive-Data/Datasets/1- GRDC-Data/South_America/stationbasins.geojson'


#folder_path = '/home/nhumair/CPSC8810-Mining-Massive-Data/Datasets/1- GRDC-Data/South_America/'

folder_path = '/home/nhumair/CPSC8810-Mining-Massive-Data/Models/Untitled_Folder'
for lat, lon, discharge, volume,station in zip(df['lat'], df['long'], df['lta_discharge'],    df['r_volume_yr'],df['grdc_no']):
    file_path= folder_path+str(station)+'_Q_Day.Cmd.txt'
    
    if(os.path.isfile(file_path)):
        print(file_path)
        print("\n")
        fields = ['YYYY-MM-DD', 'Value']
        time_series =pd.read_csv(file_path,sep=';',skipinitialspace=True,comment='#',usecols=fields,na_values=["-999"])
        time_series = time_series.dropna()
        time_series['YYYY-MM-DD'] = pd.to_datetime(time_series['YYYY-MM-DD'])
        time_series.set_index('YYYY-MM-DD', inplace=True)
        
        
        
        #fdf = pd.read_csv(file_path+'1104150_Q_Day.Cmd.txt', sep=';',skipinitialspace=True,comment='#',usecols=fields)
        #fdf = fdf.dropna()
        #fdf['YYYY-MM-DD'] = pd.to_datetime(fdf['YYYY-MM-DD'])
        #fdf.set_index('YYYY-MM-DD', inplace=True)

        scatter = vincent.Line(time_series, height=100, width=200)
        scatter.axis_titles(y='River Discharge', x='Date')
        scatter.legend(title='Station:'+str(station))
        data = json.loads(scatter.to_json())

        
        
        encoded = base64.b64encode(open("/home/nhumair/CPSC8810-Mining-Massive-Data/Models/lstm.png", 'rb').read())



        html = '<img src="data:image/png;base64,{}">'.format
        #print(20*'-',encoded.decode('UTF-8'))
        iframe = IFrame(html(encoded.decode('UTF-8')), width=100+20, height=200+20)
        popup_f = folium.Popup(iframe, max_width=450)
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            popup = popup_f,       
            color='#3186cc',
            fill=True,
            fill_color='#3186cc'
            ).add_to(us_map_2)
        
        #folium.CircleMarker(
         #   [lat, lon],
         #   radius=5,
          #  popup = Popup(max_width=450).add_child(features.Vega(data, width='100%', height='100%')),       
           # color='#3186cc',
            #fill=True,
            #fill_color='#3186cc'
            #).add_to(us_map_2)
        #print("end of loop")
        
        #folium.CircleMarker(
         #   [lat, lon],
          #  radius=5,
           # popup = ( 'Station: '+ str(station) + '<br>'
            #     'Discharge: ' + str(discharge).capitalize() + '<br>'
             #    'Volume: ' + str(volume) + '<br>'
              #   'Lat lon: ' + str(lat) +str(lon)
               #),       
            #color='#3186cc',
            #fill=True,
            #fill_color='#3186cc'
            #).add_to(us_map_2)
         
#print("finished\n")
encoded = base64.b64encode(open("/home/nhumair/CPSC8810-Mining-Massive-Data/Models/lstm.png", 'rb').read())



html = '<img src="data:image/png;base64,{}">'.format
#print(20*'-',encoded.decode('UTF-8'))
iframe = IFrame(html(encoded.decode('UTF-8')), width=100+20, height=200+20)
popup_f = folium.Popup(iframe, max_width=450)
folium.CircleMarker(
       [36.02, 0.27],
       radius=5,
       popup = popup_f,       
       color='#3186cc',
       fill=True,
       fill_color='#3186cc'
       ).add_to(us_map_2)
"""
encoded = base64.b64encode(open("/home/nhumair/CPSC8810-Mining-Massive-Data/Models/lstm.png", 'rb').read())



html = '<img src="data:image/png;base64,{}">'.format
#print(20*'-',encoded.decode('UTF-8'))
iframe = IFrame(html(encoded.decode('UTF-8')), width=632+20, height=420+20)
popup = folium.Popup(iframe, max_width=450)
for lat, lon, elevation,station in zip(df_2['LATITUDE'], df_2['LONGITUDE'], df_2['ELEVATION_(M)'],df_2['STATION_ID']):
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup = popup,
        #popup = ( 'Station: '+ str(station) + '<br>'
         #        'Elevation: ' + str(elevation).capitalize() + '<br>'
          #       'Lat lon: ' + str(lat) +str(lon)
          #     ),       
        color='red',
        fill=True,
        fill_color='red'
        ).add_to(us_map_2) 
   
folium.GeoJson(
    station_basins,
    name='geojson'
).add_to(us_map_2)
folium.TopoJson(
    station_basins,
    'objects.antarctic_ice_shelf',
    name='topojson'
).add_to(m)

folium.LayerControl().add_to(us_map_2)
"""
us_map_2.save('Features_2.html')    
