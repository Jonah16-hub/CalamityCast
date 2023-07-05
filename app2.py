import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data():
    data=pd.read_csv('GlobalTemperatures.csv')
    countries=pd.read_csv('GlobalLandTemperaturesByCountry.csv')
    continent_map=pd.read_csv('continents2.csv')
    
    return data, countries, continent_map

def process_data(data, countries, continent_map):
    # Your data processing code here ...
    # Remember to return your processed dataframes 'region' and 'countries'
    
    copy=data.copy()
    data.dropna(axis=0,inplace=True)
    data['Date']=pd.to_datetime(data.dt,errors='coerce')
    data2=data.copy()
    data2.drop(columns=['dt'],axis=1,inplace=True)
    data2['month']=data2['Date'].dt.month
    data2['year']=data2['Date'].dt.year
    data2['week']=data2['Date'].dt.isocalendar().week
    earth_data=data2.groupby(by='year')[['LandAverageTemperature', 'LandAverageTemperatureUncertainty','LandMaxTemperature', 'LandMaxTemperatureUncertainty','LandMinTemperature', 'LandMinTemperatureUncertainty','LandAndOceanAverageTemperature','LandAndOceanAverageTemperatureUncertainty']].mean().reset_index()

    # Read countries data
    countries['Date']=pd.to_datetime(countries['dt'])
    countries['Year']=countries['Date'].dt.year
    by_year=countries.groupby(by=['Year','Country']).mean().reset_index()

    #append continent
    mymap=pd.DataFrame({'Country':continent_map['name'],'Region':continent_map['region'],'alpha3':continent_map['alpha-3']})

    #merge the map to the by_year
    data=pd.merge(left=by_year,right=mymap,on='Country',how='left')
    region=data.dropna(axis=0).groupby(by=['Region','Year']).mean().reset_index()
    countries=data.dropna(axis=0).groupby(by=['Region','Country','Year']).mean().reset_index()

    return region, countries

def create_plots(region, countries):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Temperature Trends by Region", "Mean and Max Temperatures by Region"))

    colors = ['rgb(107,142,35)', 'rgb(70,130,180)', 'rgb(188,143,143)', 'rgb(222,184,135)', 'rgb(255,165,0)', 'rgb(255,0,0)']

    # line chart
    fig.add_trace(go.Scatter(x = region[region['Region'] == 'Africa']['Year'], y = region[region['Region'] == 'Africa']['AverageTemperature'], mode = 'lines',
                        name = 'Africa', marker_color=colors[0]), row = 1, col = 1)
    
    fig.add_trace(go.Scatter(x = region[region['Region'] == 'Oceania']['Year'], y = region[region['Region'] == 'Oceania']['AverageTemperature'], mode = 'lines',
                        name = 'Oceania', marker_color=colors[1]), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = region[region['Region'] == 'Americas']['Year'], y = region[region['Region'] == 'Oceania']['AverageTemperature'], mode = 'lines',
                        name = 'Oceania', marker_color=colors[2]), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = region[region['Region'] == 'Asia']['Year'], y = region[region['Region'] == 'Oceania']['AverageTemperature'], mode = 'lines',
                        name = 'Oceania', marker_color=colors[3]), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = region[region['Region'] == 'Europe']['Year'], y = region[region['Region'] == 'Oceania']['AverageTemperature'], mode = 'lines',
                        name = 'Oceania', marker_color=colors[4]), row = 1, col = 1)                   
    # bar chart
    y1 = np.round(region.groupby(by = 'Region')['AverageTemperature'].mean().tolist(), 1)
    y2 = np.round(region.groupby(by = 'Region')['AverageTemperature'].max().tolist(), 1)
    
    fig.add_trace(go.Bar(x = region['Region'].unique(), y = region.groupby(by = 'Region')['AverageTemperature'].mean().tolist(), 
                        name = 'Mean Temp', marker_color = 'rgb(188,143,143)',text = y1, textposition = 'auto'),
              row = 1, col = 2)
    fig.add_trace(go.Bar(x = region['Region'].unique(), y = region.groupby(by = 'Region')['AverageTemperature'].max().tolist(), 
                     name = 'Max Temp', marker_color = 'rgb(222,184,135)', text = y2, textposition = 'auto'),
              row = 1, col = 2)
    
    fig.show()

    # Figure layout for new plot
    fig = go.Figure()
    fig.update_layout(title='Max Temperature vs Year: 1825-2010',title_font_size=20,font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),
                      template = "ggplot2", hovermode= 'closest')
    fig.update_xaxes(showline=True,linewidth=1,linecolor='gray')
    fig.update_yaxes(showline=True,linewidth=1,linecolor='gray')

    # add trace
    for i in region['Region'].unique():
        fig.add_trace(go.Scatter(x=region[region['Region']==i]['Year'],y=region[region['Region']==i]['AverageTemperature'],mode='lines',name=i))

def main():
    data, countries, continent_map = load_data()
    region, countries = process_data(data, countries, continent_map)
    create_plots(region, countries)
    st.write(countries)

if __name__ == "__main__":
    main()




