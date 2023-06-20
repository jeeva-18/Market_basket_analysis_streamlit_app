from pyparsing import col
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
import warnings 

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from wordcloud import WordCloud

import calendar
import datetime as dt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.markdown(" <h3 style='text-align: center;'>Customer Segmentation:</h3>", unsafe_allow_html=True)
"""
Customer segmentation is the process of dividing customers into groups based on common characteristics so companies can market to each group effectively and appropriately.
* RFM (recency, frequency, monetary) Analysis
"""
"""
**RFM (recency, frequency, monetary) Analysis**

The “RFM” in RFM analysis stands for recency, frequency and monetary value. RFM analysis is a way to use data based on existing customer behavior to predict how a new customer is likely to act in the future.

1. How recently a customer has transacted with a brand
2. How frequently they’ve engaged with a brand
3. How much money they’ve spent on a brand’s products and services

RFM analysis enables marketers to increase revenue by targeting specific groups of existing customers (i.e., customer segmentation) with messages and offers that are more likely to be relevant based on data about a particular set of behaviors.
"""

@st.cache_data(experimental_allow_widgets =False, ttl= 1200)
def load_data():
    """
    This fuction loads data from the aws rds mysql table
    """
    data = None
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

    try:
        query = f"SELECT * FROM MBA_Online_Retail_Data"
        data = pd.read_sql(query,engine)

    except Exception as e:
        print(str(e))
    
    return data
#loading the data
df = load_data() 

@st.cache_data(experimental_allow_widgets =False)
def choose_country(country = "All", data = df):
  """
  This fuction takes in a country name and filters the data frame for just country
  if the there is no country inputed the fuction return the un filtered dataframe
  """
  if country == "All":
    return data
  else:
    temp_df = data[data["Country"] == country]
    temp_df.reset_index(drop= True, inplace= True)

    return temp_df

rfm_country_list = [
                    'United Kingdom',
                    'Germany',
                    'France',
                    'EIRE',
                    'Spain',
                    'Netherlands',
                    'Switzerland',
                    'Belgium',
                    'Portugal',
                    'Australia']

col1, col2, col3= st.columns((3))
with col1:
    option = st.selectbox(
        'Please Choose a country for the Recency, Frequency, Monetary Analysis',
        rfm_country_list)
    if option == "All":
        st.markdown("We will at data from All the countries")
    else:
        st.markdown(f"We will be looking at data from {option}")


RFM_df = choose_country(country=option)

#the first thing that we are going to need is the reference date 
#in this case the day after the last recorded date in the dataset plus a day
ref_date = RFM_df['InvoiceDate'].max() + dt.timedelta(days=1)
df_temp = RFM_df[RFM_df['CustomerID'] != "Guest Customer"]
RFM_df = df_temp.groupby('CustomerID').agg({'InvoiceDate': lambda x: (ref_date - x.max()).days,
                                    'InvoiceNo': lambda x: x.nunique(),
                                    'Sales Revenue': lambda x: x.sum()})

RFM_df.columns = ['Recency', 'Frequency', 'Monetary']
RFM_df["R"] = pd.qcut(RFM_df['Recency'].rank(method="first"), 4, labels=[4, 3, 2, 1])
RFM_df["F"] = pd.qcut(RFM_df['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
RFM_df["M"] = pd.qcut(RFM_df['Monetary'].rank(method="first"), 4, labels=[1, 2, 3, 4])
RFM_df['RFM_Score'] = (RFM_df['R'].astype(int)+RFM_df['F'].astype(int)+RFM_df['M'].astype(int))

RFM_df.reset_index(inplace=True)

with st.spinner("RFM analysis enables marketers to increase revenue by targeting specific groups of existing customers"):
    st.dataframe(RFM_df.head(10))
scaler = StandardScaler()
RFM_df_log = RFM_df[['Recency','Frequency','Monetary','RFM_Score']]
RFM_df_scaled = scaler.fit_transform(RFM_df_log)
RFM_df_scaled = pd.DataFrame(RFM_df_scaled)
RFM_df_scaled.columns = ['Recency','Frequency','Monetary','RFM_Score']

kmeans = KMeans(n_clusters=4, init='k-means++',n_init=10,max_iter=50,verbose=0)
kmeans.fit(RFM_df_scaled)

RFM_df['Clusters'] = kmeans.labels_
    
fig = px.box(x= RFM_df['Clusters'],y= RFM_df['Recency'],title="Clusters v Recency")
# fig.show()
st.plotly_chart(fig)


fig = px.box(x= RFM_df['Clusters'],y= RFM_df['Frequency'],title="Clusters v Frequency")
# fig.show()
st.plotly_chart(fig)

fig = px.box(x= RFM_df['Clusters'],y= RFM_df['Monetary'],title="Clusters v Monetary")
# fig.show()
st.plotly_chart(fig)

specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("RFM_Score", "Recency", "Frequency","Monetary"),
                    specs=specs
                    )

fig.add_trace(
    go.Pie(values = temp_df['RFM_Score mean'], labels = temp_df.index,
    name = 'RFM_Score'),
    1, 1
)
fig.add_trace(
    go.Pie(values = temp_df['Recency mean'], labels = temp_df.index,
    name = 'Recency'),
    1, 2
)
fig.add_trace(
    go.Pie(values = temp_df['Frequency mean'], labels = temp_df.index,
    name = 'Frequency'),
    2, 1
)
fig.add_trace(
    go.Pie(values = temp_df['Monetary mean'], labels = temp_df.index,
    name = 'Monetary'),
    2, 2
)
fig.update_layout(height=800, width=1200, title_text=" ")
st.plotly_chart(fig)




