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



