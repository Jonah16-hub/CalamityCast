#!/usr/bin/env python
# coding: utf-8

# In[2]:


!pip install geopy
!pip install plotly

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim

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

import re
# Remove non-numerical characters from Latitude column
processed_data['Latitude'] = processed_data['Latitude'].apply(lambda x: re.sub(r'[^0-9.-]', '', str(x)))
# Remove non-numerical characters from Longitude column
processed_data['Longitude'] = processed_data['Longitude'].apply(lambda x: re.sub(r'[^0-9.-]', '', str(x)))

# Display the updated dataset
print(processed_data)




# In[40]:


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
plot_variable = plt.gcf()

# In[41]:


import plotly.express as px

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
fig.show()


# In[42]:


import pandas as pd
import plotly.express as px
import geopy
from geopy.geocoders import Nominatim
# Group the data by year and disaster type and count the number of occurrences
disasters_by_year_type = processed_data.groupby(['Year', 'Disaster Type']).size().unstack().reset_index()

# Melt the dataframe to convert columns into long format
disasters_by_year_type_melted = disasters_by_year_type.melt(id_vars='Year', var_name='Disaster Type', value_name='Count')

# Create a world map plot using Plotly Express
fig = px.scatter_geo(processed_data, lat='Latitude', lon='Longitude', color='Disaster Type',
                     hover_name='Disaster Type', projection='natural earth')

# Customize the chart appearance
fig.update_layout(
    title='Natural Disasters Worldwide',
    geo=dict(showcountries=True)
)

# Show the chart
fig.show()

# In[45]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset for earthquakes
data = pd.read_csv('earthquakes_datas.csv')

# Split the data into features (X) and target variable (y)
X = data[['latitude', 'longitude', 'depth', 'mag', 'magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']]
y = data['type']

# Convert categorical variables to numerical representation
label_encoder = LabelEncoder()
X['magType'] = label_encoder.fit_transform(X['magType'])

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent' depending on your preference
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
print("Accuracy:", accuracy)


# In[46]:


from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)

# In[47]:


from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(cm)


# In[48]:


importances = model.feature_importances_
feature_names = ['latitude', 'longitude', 'depth', 'mag', 'magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']

# Sort feature importances in descending order
sorted_indices = importances.argsort()[::-1]
sorted_importances = importances[sorted_indices]
sorted_features = [feature_names[i] for i in sorted_indices]

# Print the feature importance ranking
print("Feature Importance:")
for feature, importance in zip(sorted_features, sorted_importances):
    print(f"{feature}: {importance}")


# In[50]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data2 = pd.read_csv('earthquakes_datas.csv')

# Split the data into features (X) and target variable (y)
X = data2[['latitude', 'longitude', 'depth', 'magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']]
y = data2['mag']

# Convert categorical variables to numerical representation
X = pd.get_dummies(X, columns=['magType'])

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent' depending on your preference
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('earthquakes_datas.csv')

# Split the data into features (X) and target variable (y)
X = data[['latitude', 'longitude', 'depth', 'magType', 'nst', 'gap', 'dmin', 'rms', 'horizontalError', 'depthError', 'magError', 'magNst']]
y = data['mag']

# Convert categorical variables to numerical representation
X = pd.get_dummies(X, columns=['magType'])

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent' depending on your preference
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, predictions)
print("R-squared Score:", r2)


# In[ ]:



