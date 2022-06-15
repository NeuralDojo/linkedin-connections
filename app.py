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


#DATA PROCESSING
df_connections = pd.read_csv('Complete_LinkedInDataExport_04-15-2022/Connections.csv', skiprows=2)

#keep only relevant data
df_connections = df_connections[['First Name', 'Last Name','Company', 'Position']].dropna()


#MODEL PREPARATION