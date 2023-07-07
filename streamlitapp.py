import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim
import plotly.express as px
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

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

# Lire le fichier csv
df = pd.read_csv('natural-disasters.csv')

# Convertir l'année en type date
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Ajouter une nouvelle colonne qui est la somme des dégâts économiques de toutes les catastrophes
df['Total Damage'] = df.iloc[:, 3:].sum(axis=1)

# Grouper par année et sommer
grouped = df.groupby(df['Year'].dt.year)['Total Damage'].sum()

# Créer un graphique à barres
plt.figure(figsize=(10, 5))
grouped.plot(kind='bar')
plt.title('Augmentation du cout des evenements au fil des ans')
plt.xlabel('Année')
plt.ylabel('Cout total des evenements')
plt.show()
cost_year = plt.gcf()


# Show the chart
#fig.show()

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
st.header('Increasing of the disasters frequency')
st.pyplot(number_year)

# Affichez le second graphique à barres
st.header('More details about the disasters')
st.plotly_chart(fig)

# Display the chart using Streamlit
st.plotly_chart(figwm)

st.header('Increasing of the disasters frequency')
st.pyplot(cost_year)

###Making a prediction model for earthquakes###
# Load the data
datas = pd.read_csv('earthquakes_datas.csv')

# Convert the datetime column to pandas datetime format
datas['time'] = pd.to_datetime(datas['time'])

# Split the data into features (X) and target variable (y)
X = datas[['latitude', 'longitude', 'depth', 'mag', 'magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']]
y = datas['type']

# Convert categorical variables to numerical representation
label_encoder = LabelEncoder()
X['magType'] = label_encoder.fit_transform(X['magType'])

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)
st.write("The accuracy to predict an earthquake with our model is:", accuracy)
st.write("Classification Report:")
st.text_area("",
             classification_rep,
             height=400)

###Making visual display###
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

# Create a range slider to select the desired time period
date_range = st.slider('Select Date Range', datas['time'].min().date(), datas['time'].max().date(), (datas['time'].min().date(), datas['time'].max().date()))

# Convert start_date and end_date to datetime objects
start_date = pd.to_datetime(date_range[0]).date()
end_date = pd.to_datetime(date_range[1]).date()

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
