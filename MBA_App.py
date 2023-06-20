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
from st_pages import show_pages_from_config, add_page_title

# # Either this or add_indentation() MUST be called on each page in your
# app to add indendation in the sidebar
add_page_title()

show_pages_from_config()

warnings.filterwarnings('ignore')
#lets import our data from the AWS RDS MySQL DataBase
#db info
from sqlalchemy import create_engine


host = 'database-1.cujz4kbilje1.eu-north-1.rds.amazonaws.com'
user = 'admin'
password = 'Qwe12345'
port = '3306'
database = 'database-1'


st.set_option('deprecation.showPyplotGlobalUse', False)






col1, col2, col3 = st.columns((.1,1,.1))

with col1:
    st.write("")

with col2:
    st.markdown(" <h1 style='text-align: center;'>RETAIL ANALYTICS : Unveiling Consumer Behavior and Enhancing Sales through EDA,Market Basket Analysis, Customer Segmentation and Product Recommendation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><i><b>Providing a Retail Business with a strategy which helps improve their "
                "product sales, inventory management, and customer retention, which in turn would improve the profitability of the business.</b></i></p>", unsafe_allow_html=True)
    st.markdown("<center><img src='https://github.com/jeeva-18/streamlit_app/blob/main/Assets/Grey-Goods-under-the-Consumer-Protection-Act-e1566375131654.jpg' width=600/></center>", unsafe_allow_html=True)


with col3:
    st.write("")


st.markdown("----")

col1, col2,col3 = st.columns((1,0.1,1))

with col1:
   
    st.markdown("### ***Project Contributors:***")
    st.markdown("Kuzi Rusere")

    st.markdown("### **Project Introduction**")
    st.markdown("***Business Proposition:*** This project aims to provide a Retail "
                "Business with a strategy that helps improve their product sales, "
                "inventory management, and customer retention, which in turn would "
                "improve the profitability of the business. In the retail environment, "
                "profitability and the `bottom line` is at the focal point of any "
                "organization or business in this space. From product sales, through "
                "inventory management to customer acquisition and retention all this one "
                "way or the other affects the business' profits and net revenue. Transaction "
                "data from the POS (point of sale) systems for a retail business is a treasure "
                "trove of insights that can be used to better understand the products, customer "
                "purchase behavior, and sales together with the relationship and patterns in "
                "between. This project explores the different ways of deriving these insights, "
                "patterns, and relationships, and how these can be used in designing, developing, "
                "and implementing a strategy to improve the retail business' profits, revenue, "
                "and overall operations")
    st.markdown("***Methodology:*** Data Mining, Analysis and Visualization of Retail "
                "Sales Data.")
    """
    1. Market Basket Analysis (MBA), which aims to find relationship and establishing pattens within the retail sales data. 
    2. Customer Segmentation 
    > * RFM (recency, frequency, monetary) Analysis
    3. Product Recomendation (people who bought this also bought)
    """
    st.markdown("In addition, we created this `Streamlit` interactive data visualization "
                "tool that allows users interact with the data and analytics.")
with col2:
    pass
with col3:
    st.markdown("### ***Data Collection:***")

    """
    **General Information About the Data**

    This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

    **Information about the Attributes/Columns in the Dataset**
    * ***InvoiceNo:*** Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
    * ***StockCode:*** Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
    * ***Description:*** Product (item) name. Nominal.
    * ***Quantity:*** The quantities of each product (item) per transaction. Numeric.
    * ***InvoiceDate:*** Invice Date and time. Numeric, the day and time when each transaction was generated.
    * ***UnitPrice:*** Unit price. Numeric, Product price per unit in sterling.
    * ***CustomerID:*** Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
    * ***Country:*** Country name. Nominal, the name of the country where each customer resides.

    ###### **The data source:**
    
    """

    st.image("Assets/UCI_ML_REPO.png", caption="https://archive.ics.uci.edu/ml/datasets/online+retail")


st.markdown("----")

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



st.markdown("#### ***Lets take a look at the data:***")
"""
We are going to use the pandas `.shape` function/method to the total number of columns and rows of the dataframe. We can see that our dataframe contains 481313 rows and 16 columns

We'll use the pandas `.info()` function so see the general infomation (data types, null value count, etc.) about the data.
"""
st.markdown(f"###### ***The shape of the data***: {df.shape}")


col1, col2,col3 = st.columns((1, 0.01,.5))

df_head = pd.read_csv("df_head.csv")
with col1:
    st.markdown("***The below is the first 5 rows of the cleaed dataset***")
    st.dataframe(df_head)
with col2:
    pass
df_info = pd.read_csv("df_info.csv", index_col=0)
with col3:
    st.markdown("***The below is the info of the data***")
    st.dataframe(df_info)

st.success("If you want to take a look at how the data was cleaned, you "
            "can go check out the jupyter notebook of this project at: "
            "https://github.com/kkrusere/Market-Basket-Analysis-on-the-Online-Retail-Data/blob/main/MBA_Online-Retail_Data.ipynb")

######################functions############################

@st.cache_data(experimental_allow_widgets =False)
def group_Quantity_and_SalesRevenue(df,string):
    """ 
    This function inputs the main data frame and feature name 
    The feature name is the column name that you want to group the Quantity and Sales Revenue
    """

    df = df[[f'{string}','Quantity','Sales Revenue']].groupby([f'{string}']).sum().sort_values(by= 'Sales Revenue', ascending = False).reset_index()

    return df

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


def wordcloud_of_Description(df, title):
    """
    This fuction creates a word cloud
    inputs a data frame converts it to tuples and uses the input 'title' as the title of the word cloud
    """
    plt.rcParams["figure.figsize"] = (20,20)
    tuples = [tuple(x) for x in df.values]
    wordcloud = WordCloud(max_font_size=100,  background_color="white").generate_from_frequencies(dict(tuples))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontsize = 27)
    plt.show()


country_list = ["All"] + list(dict(df['Country'].value_counts()).keys())
@st.cache_data(experimental_allow_widgets =False)
def choose_country(country, data = df):
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
##################################################################################
st.markdown("---")
