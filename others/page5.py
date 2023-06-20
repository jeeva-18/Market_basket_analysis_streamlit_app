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
from numpy.lib.arraysetops import unique





st.markdown(" <h3 style='text-align: center;color:#FA5B3C''>Product recomendation <i>(people who bought this also bought)</i>:</h3>", unsafe_allow_html=True)
with st.spinner("Generating the Frequent Itemsets and Assosiation Rules..."):
  product = pd.read_csv('others/product_rec.csv')

st.dataframe(product)



