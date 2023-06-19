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

st.markdown(" <h3 style='text-align: center;'>Exploratory Data Analysis <i>(EDA)</i>:</h3>", unsafe_allow_html=True)
col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    * Exploratory data analysis is an approach/practice of analyzing data sets to summarize their main characteristics, often using statistical graphics and other ***data visualization***. It is a critical process of performing initial ***investigations to discover*** patterns, detect outliers and anomalies, and gain some new, hidden, insights into the data.
    * Investigating questions like the total volume of purchases per month, week, day of the week, time of the day right to the hour. We will look at customers more later when we get into the ***Recency, Frequency and Monetary Analysis (RFM)*** in the Customer Segmentation section of the project.
    """
with col3:
    pass
############################################
col1, col2, col3= st.columns((1,.1,1))

with col1:
    Country_Data = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(10)
    fig = px.bar(Country_Data, x= "InvoiceNo", y='Country', title= "Top 10 Number of orders per country with the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    st.markdown("UK has more number of orders witk 16k Invoice numbers")

with col2:
    pass

with col3:
    Country_Data = df[df['Country'] != "United Kingdom"].groupby("Country")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(10)
    fig = px.bar(Country_Data, x= "InvoiceNo", y='Country', title= "Top 10 Number of orders per country without the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
########################################################

col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    The above charts show that the UK by far has more invoices, just as suspected, with invoices surpassing 15K. Germany in in second place, with approximately 30 time less invoices. The retail store management can start possing question of why this is the case, especially when this is a Online retail store. Question like, what is the traffic like to the store web page, or should they start thinking of ***Search Engine Optimization (SEO)***, which is the process of improving the quality and quantity of website traffic to a website or a web page from search engines. Many other questions can be raised from the 2 charts above.

    Below, we can take a look at how the countries fare up with regards to the ***Quantity sold*** and ***Sales Revenue***.The first plot is going to be for Quantity sold and the second will be for Sales Revenue both for the whole year of 2011.
    """
with col3:
    pass

####################
col1, col2, col3= st.columns((1,.1,1))
with col1:
    #choice = st.radio("", ("Top 10", "Bottom 10"))
    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df, x= "Quantity", y='Country', title= "Quantity of orders per country with the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass

with col3:

    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df[temp_df['Country'] != "United Kingdom"], x= "Quantity", y='Country', title= "Quantity of orders per country without the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)


####################

col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    Just as expected, the UK has high volumes of Quantitly sold and the below charts should show that the UK has high sales as well. However, unlike the number of invoices, the Netherlands has the second highest volume of Quantity sold at approximately 200K. 
    """
with col3:
    pass

####################
col1, col2, col3= st.columns((1,.1,1))
with col1:
    #choice = st.radio("", ("Top 10", "Bottom 10"))
    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df, x= "Sales Revenue", y='Country', title= "Sales Revenue of orders per country with the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass

with col3:
    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df[temp_df['Country'] != "United Kingdom"], x= "Sales Revenue", y='Country', title= "Sales Revenue of orders per country without the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

####################
col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    The sales revenue of Netherlands and Germany is quite close. It would be interesting to see this broken down by time periods: 'Month', 'Week', 'Day of the Week', 'Time of Day' ,or 'Hour'.
    
    We now going to look at the products, which ones have high Quantity sold, or which product has high Sales Revenue. But first the below chart is a wordcloud of the product descriptions. A wordcloud is a visual representations of words that give greater prominence to words that appear more frequently, in this case the frequency is the 'Quantity'
    """
with col3:
    pass

####################

col1, col2, col3= st.columns((3))
with col1:
    option = st.selectbox(
        'Please Choose a country to Analyze',
        country_list)
    if option == "All":
        st.markdown("We will at data from All the countries")
    else:
        st.markdown(f"We will be looking at data from {option}")


dataframe = choose_country(country=option)


st.markdown("###### **We can create a word cloud of the Product Descriptions per Quantity & Product Descriptions per Sales Revenue**")

col1, col2, col3= st.columns((1,.3,1))
with col1:
    temp_df = pd.DataFrame(dataframe.groupby('Description')['Quantity'].sum()).reset_index()
    title = "Product Description per Quantity"
    wordcloud_of_Description(temp_df, title)
    st.pyplot()

with col2:
    pass

with col3:
    temp_df = pd.DataFrame(dataframe.groupby('Description')['Sales Revenue'].sum()).reset_index()
    title = "Product Description per Sales Revenue"
    wordcloud_of_Description(temp_df, title)
    st.pyplot()

###############################################

st.markdown("##### **Monthly Stats:**") 
"""
Below are the monthly analysis of the Sales and the Quantity of iterms sold
"""

col1, col2, col3= st.columns((1,.3,1))
with col1:
    temp_df = group_Quantity_and_SalesRevenue(dataframe,'Month')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=("Quantity", "Sales Revenue")
                    )
    fig.add_trace(go.Bar(x=temp_df['Month'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Month'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(showlegend=False, title_text="Monthly Sales Revanue and Quantity")
    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    """
    The above graphs show the monthly trend of Quantity of products ordered(left) and Sales Revenue(right).
    """

with col2:
    pass

with col3:
    fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "pie"}, {"type": "pie"}]], 
                    subplot_titles=("Quantity per Month", "Sales Revenue per Month")
                    )

    fig.add_trace(
        go.Pie(values = temp_df['Quantity'], labels = temp_df['Month'],
        name = 'Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Sales Revenue'], labels = temp_df['Month'],
        name = 'Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(title_text="Percentage pie charts for Monthly Sales Revanue and Quantity")

    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    """
    The above pie charts depicts the quantity of products ordered and sales revenue per month. 
    """

##############################
st.markdown("##### **Weekly Stats:**")
"""
The below are the weekly analysis of the Sales and the Quantity of iterms sold
"""
ccol1, col2, col3= st.columns((.5,1,.5))
with col1:
    pass
with col2:
    
    temp_df = group_Quantity_and_SalesRevenue(dataframe,'Week of the Year')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=("Quantity", "Sales Revenue")
                            )

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(showlegend=False, title_text="Weekly Sales Revanue and Quantity")
    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

    """
    The above graphs shows the weekly trend of sales revenue and the quantity of products ordered. 
    """

with col3:
    pass

##############################
st.markdown("##### **Daily Stats:**")
"""
The below are the daily analysis of the Sales and the Quantity of iterms sold
"""
col1, col2, col3= st.columns((1,.3,1))
with col1:
    temp_df = group_Quantity_and_SalesRevenue(dataframe,'Day of Week')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=("Quantity", "Sales Revenue")
                            )

    fig.add_trace(go.Bar(x=temp_df['Day of Week'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Day of Week'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(coloraxis=dict(colorscale='Greys'), showlegend=False, title_text="Day of the Week Sales Revanue and Quantity")
    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

    st.markdown("The above graphs depict the daily trend of Sales revenue and quantity.")

with col2:
    pass

with col3:
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "pie"}]], 
                        subplot_titles=("Quantity", "Sales Revenue")
                        )

    fig.add_trace(
        go.Pie(values = temp_df['Quantity'], labels = temp_df['Day of Week'],
        name = 'Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Sales Revenue'], labels = temp_df['Day of Week'],
        name = 'Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(title_text="Percentage pie charts for Day of the Week Sales Revanue and Quantity")

    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    st.markdown("The above pie charts shows the daily trend of sales revenue and quantity of products ordered.")


###############################


col1, col2, col3= st.columns((.5,1,.5))
with col1:
    pass

with col2:
    temp_df = group_Quantity_and_SalesRevenue(dataframe,'Time of Day')
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "pie"}]], 
                        subplot_titles=("Quantity", "Sales Revenue")
                        )

    fig.add_trace(
        go.Pie(values = temp_df['Quantity'], labels = temp_df['Time of Day'],
        name = 'Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Sales Revenue'], labels = temp_df['Time of Day'],
        name = 'Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(title_text="Percentage pie charts for Time of Day Sales Revanue and Quantity")

    st.plotly_chart(fig)
    st.markdown("The above piecharts shows the breakdown of Quantity of orders(left) and Sales revenue(right) by time of the day.")

with col3:
    pass

###############################

col1, col2, col3= st.columns((1,.1,1))
with col1:
    #we can also look at the volume of Invoice Numbers hourly data 
    Hourly_Sales = (dataframe.groupby('Hour').sum()["Quantity"]).reset_index()
    fig = px.bar(Hourly_Sales, x='Hour', y='Quantity', title='Hourly Volume of quantity sold')
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show(height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass

with col3:
    #we can also look at the volume quantity sold hourly data 
    Hourly_Sales = (dataframe.groupby('Hour').count()["InvoiceNo"]).reset_index()
    fig = px.bar(Hourly_Sales, x='Hour', y='InvoiceNo', title='Hourly sale using the Invoice Numbers')
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show(height=700, width=1000)
    st.plotly_chart(fig)

###############################

st.markdown("##### ***Customers:***")

col1, col2, col3= st.columns((1,.1,1))
with col1:
    data = dataframe.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(11)
    fig = px.bar(data, x='CustomerID', y='InvoiceNo', title='Graph of top ten customer with respect to the invoice number')
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show(height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass

with col3:
    temp_df = dataframe[dataframe["CustomerID"] != "Guest Customer"]
    data = temp_df.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(11)
    fig = px.bar(data, x='CustomerID', y='InvoiceNo', title='Graph of top ten customer with respect to the invoice number without the Guest Customer')
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show(height=700, width=1000)
    st.plotly_chart(fig)


#################################################


temp_df = group_Quantity_and_SalesRevenue(dataframe, 'Description')
Quantity_tempA = temp_df.sort_values(ascending=False, by = "Quantity").reset_index(drop=True)
Sales_Revenue_tempA = temp_df.sort_values(ascending=False, by = "Sales Revenue").reset_index(drop=True)

Quantity_tempA.drop('Sales Revenue', axis=1, inplace=True)
Sales_Revenue_tempA.drop('Quantity', axis=1, inplace=True)


colspace, col1, col2, col3= st.columns((.45,1,.01,1))
with col1:
    qchoice = st.radio("Choose Either Top or Bottom of Product Description by Quantity", ("Top 10", "Bottom 10"))
    qchoice_dict = {"Top 10":Quantity_tempA.head(10), "Bottom 10": Quantity_tempA.tail(10)}
    st.markdown(f"{qchoice} Description by Quantity")
    st.dataframe(qchoice_dict.get(qchoice))

with col3:
    schoice = st.radio("Choose Either Top or Bottom of Product Description by Sales Revenue", ("Top 10", "Bottom 10"))
    schoice_dict = {"Top 10":Sales_Revenue_tempA.head(10), "Bottom 10": Sales_Revenue_tempA.tail(10)}
    st.markdown(f"{schoice} Description by Sales Revenue")
    st.dataframe(schoice_dict.get(schoice))


##################################################

col1, col2, col3= st.columns((.5,1,.5))
with col1:
    pass

with col2:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=(f"{qchoice} Products by Quantity", f"{schoice} Products by Sales Revenue")
                            )

    fig.add_trace(go.Bar(x=qchoice_dict.get(qchoice)['Description'], y=qchoice_dict.get(qchoice)['Quantity'],name = f"{qchoice}"),1, 1)

    fig.add_trace(go.Bar(x=schoice_dict.get(schoice)['Description'], y=schoice_dict.get(schoice)['Sales Revenue'],name = f"{schoice}"),1, 2)
    fig.update_layout(height=600)
    fig.update_layout(showlegend=False, title_text="Product Description by Quantity and Sales Revenue")
    #fig.show(renderer='png', height=700, width=1200)
    #fig.show(height=700, width=1000)
    st.plotly_chart(fig)

with col3:
    pass

#####################################################












#####################################################

st.markdown("----")




