from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
import matplotlib as plt
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import random
from PIL import Image

st.set_page_config(page_title = 'Similar Linkeding Connections',
                    layout="wide",
                    menu_items={
                                'About': "# This is a header. This is an *extremely* cool app!"
                    }
)

st.title("Similar linkedin connections with sentence-transformers")
#INSERTING LOGO
img = Image.open('img/linkedin-statistics.jpeg')
st.image(img, width=800)


#DATA IMPORT & PROCESSING
df_connections = pd.read_csv('data/lkd_connections.csv', skiprows=2)
df_2d_embeddings = pd.read_csv('data/2d_lkd_connections.csv')
df_3d_embeddings = pd.read_csv('data/3d_lkd_connections.csv')

#FILTERS

add_selectCluster = st.sidebar.select_slider(
    'Select number of clusters',
    options = list(range(1,21)),
    value=20
)

# Add a selectbox to the sidebar:
add_selectPositionBox = st.sidebar.multiselect(
    'Select people positions',
    df_2d_embeddings.Position.unique()
)

if (len(add_selectPositionBox)==0):
    data_fig = df_3d_embeddings
else:
    data_fig = df_3d_embeddings.query("Position==@add_selectPositionBox")

add_selectColor = st.sidebar.selectbox(
    'Select color scale:',
    options=px.colors.named_colorscales(),
)

#CLUSTERING
#clustering with k-means
NUM_CLUSTERS = add_selectCluster

from sklearn.cluster import KMeans

clusters = KMeans(n_clusters=NUM_CLUSTERS)
clusters.fit(df_2d_embeddings[['x','y']])

clusters_3d = KMeans(n_clusters=NUM_CLUSTERS)
clusters_3d.fit(data_fig[['x','y','z']])

df_2d_embeddings['clusters'] = clusters.labels_
#df_2d_embeddings['company'] = df_connections.Company
data_fig['clusters'] = clusters_3d.labels_
#df_3d_embeddings['company'] = df_connections.Company

#FIGURES

RAND_COLOR = add_selectColor

fig_2d = px.scatter(data_frame = df_2d_embeddings, x ='x', y='y', hover_name='Position', 
                 hover_data=[df_2d_embeddings.index], color = 'clusters', opacity=0.3,
                 template='plotly_dark',
                 color_continuous_scale = 'Blues',
                 width = 1000, height = 800)


fig_3d = px.scatter_3d(data_frame = data_fig, x ='x', y='y', z='z', hover_name ='Position', 
                 hover_data=[data_fig.index], color = 'clusters', opacity=0.2,
                 color_continuous_scale=RAND_COLOR, template='plotly_dark',
                 title="3D Plot Linkedin Connections with "+str(NUM_CLUSTERS)+" clusters",
                 width = 1000, height = 700)

fig_3d.update_traces(marker={'size': 3})


##BUILDING VIZ REPORT

st.write("Some basic data:")

st.markdown("- Number of Connections: "+str(df_connections.shape[0]))

#st.plotly_chart(fig_2d)
st.plotly_chart(fig_3d)