import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

# Configuration de l'interface de Streamlit
st.set_page_config(page_title='CalamityCast', 
                   page_icon=':warning:', 
                   layout='wide', 
                   initial_sidebar_state='expanded')

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

# Les données pour les graphiques
# Remplacez par vos vraies données
data1 = pd.Series(np.random.rand(10), index=list('ABCDEFGHIJ'))
data2 = pd.Series(np.random.rand(10), index=list('ABCDEFGHIJ'))

# Création de la page Streamlit

# Affichez le titre et une brève description
st.title('CalamityCast')
st.write("""
Solution logiciel d'aide à la décision face aux évènements climatiques extrèmes.
""")

# Affichez le premier graphique à barres
st.header('Premier graphique à barres')
st.pyplot(create_bar_chart(data1))

# Affichez la carte
st.header('Carte Interactive')
m = create_map()
folium_static(m)

# Affichez le second graphique à barres
st.header('Deuxième graphique à barres')
st.pyplot(create_bar_chart(data2))

# Pour personnaliser le thème de votre application, vous devez ajouter un fichier .streamlit/config.toml à votre projet avec le contenu suivant :
"""
[theme]
primaryColor = "#008000"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F0F0"
textColor = "#111111"
font = "sans-serif"
"""
# Notez que vous devez remplacer les valeurs hexadécimales de couleur par celles de votre choix.
