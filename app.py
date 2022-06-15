import pandas as pd
import matplotlib as plt
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
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
df_2d_embeddings = pd.read_csv('data/2d_lk_connections.csv')
df_3d_embeddings = pd.read_csv('data/3d_lk_connections.csv')


#CLUSTERING
#clustering with k-means
NUM_CLUSTERS = 20

from sklearn.cluster import KMeans

clusters = KMeans(n_clusters=NUM_CLUSTERS)
clusters.fit(df_2d_embeddings[['x','y']])

clusters_3d = KMeans(n_clusters=NUM_CLUSTERS)
clusters_3d.fit(df_3d_embeddings[['x','y','z']])

df_2d_embeddings['clusters'] = clusters.labels_
df_2d_embeddings['company'] = df_connections.Company
df_3d_embeddings['clusters'] = clusters.labels_
df_3d_embeddings['company'] = df_connections.Company