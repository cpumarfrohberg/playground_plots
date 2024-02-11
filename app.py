#app.py
#TODO: modify data => change col names and drop unnecessary ones
from explore_data.visualizer import Visualizer
from explore_data.reader import Reader
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

CATS = [
    'Resident_Iquitos', 'Client_Sex', 'Client_Married', 'Client_Status_Post3Months', 
    'LO_Active_Employee_Post3Months', 'LO_Active_Employee_Prior3Months', 
    'LO_Active_Employee_Prior6Months', 'LO_Changed'
    ]

viz = Visualizer()
reader = Reader()

@st.cache_data
def load_data():
    '''Read raw data and return DataFrame(s).'''
    lo_client_relation = Reader.read_data('Fahrraddiebstahl')
    churn_data = Reader.read_data('churn_data')
    return lo_client_relation, churn_data

lo_client_relation, churn_data = load_data()

st.title('Visual Analyis')

nav = st.sidebar.radio(
    'Please chose one of the following:',
    ['Home', 'Categorical Variables', 'Numeric Variables', 'Time Series', 'Heat Maps'] #TODO: adapt
    ) 

if nav == 'Home':
    st.markdown(
    ''' ## This is an example on how to visualize tabular data and on how to make initial inferences.
    '''
    )
  
    if st.checkbox('<- For a first scan of data related to the relationship between Loan Officers and \
                   clients, click here'):
            st.table(lo_client_relation)
    
    if st.checkbox('<- For a first scan of data on customer churn, click here'):
            st.table(churn_data)

if nav == 'Categorical Variables':
    st.write('Plot and make initial inference on categorical variables.')
    if st.checkbox('<- Click here for example plots of categorical variables'): #TODO: implement as method
        fillable_plots = list()
        for cat in CATS:
            fig, ax = plt.subplots(figsize = (5,3))
            fig = sns.catplot(
                    data=churn_data, #TODO: check dataset name
                    y= cat, 
                    kind='count', 
                    alpha = 0.5, 
                    edgecolor='.6'
            )
            fillable_plots.append(fig)
        for plot in fillable_plots:
           time.sleep(3)
           st.pyplot(plot)
    
if nav == 'Numeric Variables':
    st.write('Plot and make initial inference on numeric variables.')
    if st.checkbox('<- Click here for inspecting box plots'):
        boxplots = viz.boxplot(columns=lo_client_relation.columns, y_max=None) #TODO: check dataset name
        try:
            st.pyplot(boxplots)
        except: 'StreamlitMethodError'

    if st.checkbox('<- Click here for checking pairplots and scatterplots'):
        pairplot = viz.pairplot()
        try:
            st.pyplot(pairplot());
        except: 'StreamlitMethodError'
    
    if st.checkbox('<- Click here for checking QQplots (assumption: vars are normally distributed)'):
        qqplot = viz.qq_plot()
        try:
            st.pyplot(qqplot);
        except: 'StreamlitMethodError'

if nav == 'Plot thru time':
    st.write('Welcome to the section on time series.')

    if st.checkbox('<- Click here to see the daily values of bike thefts.'):
        bike_theft_series = churn_data.loc['2022-01-02':'2023-02-19'].resample('D').size()
        st.line_chart(bike_theft_series) 
    
    if st.checkbox('<- Click here to see the weekly values of bike thefts.'):
        bike_theft_series = churn_data.loc['2022-01-02':'2023-02-19'].resample('W').size()
        st.line_chart(bike_theft_series) 

if nav == 'Heat Maps':
    st.markdown(
    ''' #### Welcome to the heat map page.
    '''
    )
      
    if st.checkbox('Click here to see the how variables are correlated with each other (pearson).'):
        corr = churn_data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, ax=ax)
        st.write(fig)