import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import pandas as pd
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim
import plotly.express as px
import re

#################################################
#   Configuration de l'interface de Streamlit   #
#################################################
st.set_page_config(page_title='CalamityCast', 
                   page_icon='CalamityCastLogo-02.png', 
                   layout='wide', 
                   initial_sidebar_state='expanded')

#################################################
#                DATA PREPROCESSING             #
#################################################
# Load the dataset into a Pandas DataFrame
data = pd.read_excel('emdat_public_2023_05_02_query_uid-umjMCi.xlsx')

# Convert the 'Longitude' and 'Latitude' columns to strings
data['Longitude'] = data['Longitude'].astype(str)
data['Latitude'] = data['Latitude'].astype(str)

selected_columns = ['Dis No', 'Year', 'Disaster Type', 'Disaster Subgroup', 'Disaster Subtype', 'Country', 'Longitude', 'Latitude']

def select_columns(dataset):
    return dataset[selected_columns]

def complete_coordinates(dataset):
    geolocator = Nominatim(user_agent="your_app_name")  # Initialize the geocoding service
    
    for index, row in dataset.iterrows():
        if pd.isnull(row['Latitude']) or pd.isnull(row['Longitude']):
            try:
                location = geolocator.geocode(query=row['Country'])
                if location is not None:
                    dataset.at[index, 'Latitude'] = location.latitude
                    dataset.at[index, 'Longitude'] = location.longitude
            except geopy.exc.GeocoderTimedOut:
                # Handle timeout exception, if necessary
                continue

    return dataset


selected = select_columns(data)
processed_data = complete_coordinates(selected)

# Remove non-numerical characters from Latitude column
processed_data['Latitude'] = processed_data['Latitude'].apply(lambda x: re.sub(r'[^0-9.-]', '', str(x)))
# Remove non-numerical characters from Longitude column
processed_data['Longitude'] = processed_data['Longitude'].apply(lambda x: re.sub(r'[^0-9.-]', '', str(x)))


#################################################
#                 Starting dataviz              #
#################################################
# Group the data by year and count the number of occurrences
disasters_by_year = processed_data.groupby('Year').size()

# Create a bar chart or histogram
plt.figure(figsize=(12, 8))
plt.bar(disasters_by_year.index, disasters_by_year.values)

# Customize the chart appearance
plt.xlabel('Year')
plt.ylabel('Number of Disasters')
plt.title('Number of Disasters by Year')

# Show the histogram
plt.show()
# Save plot 
number_year = plt.gcf()

# Group the data by year and disaster type and count the number of occurrences
disasters_by_year_type = data.groupby(['Year', 'Disaster Type']).size().unstack().reset_index()

# Melt the dataframe to convert columns into long format
disasters_by_year_type_melted = disasters_by_year_type.melt(id_vars='Year', var_name='Disaster Type', value_name='Count')

# Create the stacked bar chart using Plotly Express
fig = px.bar(disasters_by_year_type_melted, x='Year', y='Count', color='Disaster Type', barmode='stack')

# Customize the chart appearance
fig.update_layout(
    title='Number of Disasters by Year and Type',
    xaxis_title='Year',
    yaxis_title='Number of Disasters'
)

# Show the chart
#fig.show()

# Fonction pour créer une carte avec Folium
def create_map():
    m = folium.Map(location=[48.85, 2.35], zoom_start=6)
    # Ajoutez ici d'autres fonctionnalités folium, comme des marqueurs ou des calques de chaleur.
    return m

# Fonction pour créer des graphiques
def create_bar_chart(data):
    fig, ax = plt.subplots()
    ax.bar(data.index, data.values)
    return fig

##World map graph##
# Create a world map plot using Plotly Express
figwm = px.scatter_geo(processed_data, lat='Latitude', lon='Longitude', color='Disaster Type',
                     hover_name='Disaster Type', projection='natural earth')

# Customize the chart appearance
figwm.update_layout(
    title='Natural Disasters Worldwide',
    geo=dict(showcountries=True)
)


#################################################
#          Création de la page Streamlit        #
#################################################

# Affichez le titre et une brève description
st.title('CalamityCast')
st.write("""
Solution logiciel d'aide à la décision face aux évènements climatiques extrèmes.
""")

# Affichez le premier graphique à barres
st.header('Premier graphique à barres')
st.pyplot(number_year)

# Affichez la carte
st.header('Carte Interactive')
m = create_map()
folium_static(m)

# Affichez le second graphique à barres
st.header('Deuxième graphique à barres')
st.plotly_chart(fig)

# Display the chart using Streamlit
st.plotly_chart(figwm)




# Load the data
datas = pd.read_csv('earthquakes_datas.csv')

# Convert the datetime column to pandas datetime format
datas['time'] = pd.to_datetime(datas['time'])

# Create a world map plot using Plotly Express
figwm2 = px.scatter_geo(datas, lat='latitude', lon='longitude', color='type',
                        hover_name='place', projection='natural earth')

# Customize the chart appearance
figwm2.update_layout(
    title='Natural Disasters Worldwide',
    geo=dict(showcountries=True)
)

# Add an empty placeholder at the bottom to push the content up
st.empty()

# Create a date range slider to select the desired time period
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input('Start Date', datas['time'].min().date())
with col2:
    end_date = st.date_input('End Date', datas['time'].max().date())

# Convert start_date and end_date to datetime objects
start_date = pd.to_datetime(start_date).date()
end_date = pd.to_datetime(end_date).date()

# Filter the data based on the selected date range
filtered_data = datas[(datas['time'].dt.date >= start_date) & (datas['time'].dt.date <= end_date)]

# Create a world map plot using Plotly Express
figwm2_filtered = px.scatter_geo(filtered_data, lat='latitude', lon='longitude', color='type',
                                 hover_name='place', projection='natural earth')

# Customize the chart appearance
figwm2_filtered.update_layout(
    title='Natural Disasters Worldwide - {} to {}'.format(start_date, end_date),
    geo=dict(showcountries=True)
)

st.plotly_chart(figwm2_filtered)


# Pour personnaliser le thème de votre application, vous devez ajouter un fichier .streamlit/config.toml à votre projet avec le contenu suivant :
"""

"""
# Notez que vous devez remplacer les valeurs hexadécimales de couleur par celles de votre choix.
