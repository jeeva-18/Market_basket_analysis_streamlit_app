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
  
  st.markdown(" <h3 style='text-align: center;'>Market Basket Analysis <i>(MBA)</i>:</h3>", unsafe_allow_html=True)
r"""
**What is Market Basket Analysis?:**

Market Basket Analysis (MBA) is a data mining technique that is mostly used in the Retail Industry to uncover customer purchasing patterns and product relationships. The techniques used in MBA identify the patterns, associations, and relationships (revealing product groupings and which products are likely to be purchased together) in in frequently purchased items by customers in large transaction datasets collected/registered at the point of sale. The results of the Market Basket Analysis can be used by retailers or marketers to design and develop marketing and operation strategies for a retail business or organization.<br>
Market basket analysis mainly utilize Association Rules {IF} -> {THEN}. However, MBA assigns Business outcomes and scenarios to the rules, for example,{IF X is bought} -> {THEN Y is also bought}, so X,Y could be sold together. <br>

Definition: **Association Rule**

Let $I$= \{$i_{1},i_{2},\ldots ,i_{n}$\} be an itemset.

Let $D$= \{$t_{1},t_{2},\ldots ,t_{m}$\} be a database of transactions $t$. Where each transaction $t$ is a nonempty itemset such that ${t \subseteq I}$

Each transaction in D has a unique transaction ID and contains a subset of the items in I.

A rule is defined as an implication of the form:
$X\Rightarrow Y$, where ${X,Y\subseteq I}$.

The rule ${X \Rightarrow Y}$ holds in the dataset of transactions $D$ with support $s$, where $s$ is the percentage of transactions in $D$ that contain ${X \cup Y}$ (that is the union of set $X$ and set $Y$, or, both $X$ and $Y$). This is taken as the probability, ${P(X \cup Y)}$. Rule ${X \Rightarrow Y}$ has confidence $c$ in the transaction set $D$, where $c$ is the percentage of transactions in $D$ containing $X$ that also contains $Y$. This is taken to be the conditional probability, like ${P(Y | X)}$. That is,

* support ${(X \Rightarrow Y)}$ = ${P(X \cup Y)}$

* confidence ${(X \Rightarrow Y)}$ = ${P(X|Y)}$

The lift of the rule ${(X \Rightarrow Y)}$  is the confidence of the rule divided by the expected confidence, assuming that the itemsets $X$ and $Y$ are independent of each other.The expected confidence is the confidence divided by the frequency of ${Y}$.

* lift ${(X \Rightarrow Y)}$ = ${ \frac {\mathrm {supp} (X\cap Y)}{\mathrm {supp} (X)\times \mathrm {supp} (Y)}}$


Lift value near 1 indicates ${X}$ and ${Y}$ almost often appear together as expected, greater than 1 means they appear together more than expected and less than 1 means they appear less than expected.Greater lift values indicate stronger association

"""

"""



"""

"""
##### ***Now the Implementation of the MBA***
"""
mba_country_list = [
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


"""
We are going to use the Apriori Algorithm for the association rule mining/analysis. Apriori is an algorithm for frequent item set mining and association rule learning over relational dataset. It proceeds by identifying the frequent individual items in the dataset and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the dataset. The frequent item sets determined by Apriori can be used to determine association rules which highlight general trends, pattern, and relationships in the dataset.
"""
"""
'The next step will be to generate the frequent itemsets that have a support of at "
"least 10% using the MLxtend Apriori fuction which returns frequent itemsets from a "
"one-hot DataFrame. And then can look at the rules  of association using the "
"`MLxtend association_rules(), The function generates a DataFrame of association "
"rules including the metrics 'score', 'confidence', and 'lift'
"""
with col2:
    with st.spinner("Generating the Frequent Itemsets and Assosiation Rules..."):
        rules = pd.read_csv('rules.csv')

    """Assosiation Rules"""
    st.dataframe(rules.head())
with col3:
    pass






st.markdown("----")
