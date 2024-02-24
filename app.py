#app.py
from explore_data.visualizer import Visualizer
from explore_data.reader import Reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import StringIO

url = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'

@st.cache_data
def load_data():
    '''Read raw data and return DataFrame(s).'''
    reader = Reader()
    cars = reader.read_data(url)
    return cars

def instantiate_vizualizer():
    '''Read raw data and return DataFrame(s).'''
    car_viz = Visualizer(cars)
    return car_viz

cars = load_data()
car_viz = instantiate_vizualizer()

numeric = cars.select_dtypes(include=['int64', 'float64'])

st.title('Example Visual Analyis')

st.markdown(
    ''' #### This is an example on how to visualize tabular data and on how to make initial inferences.
    '''
    )

nav = st.sidebar.radio(
    'Please chose one of the following:',
    ['Home', 'Categorical Variables', 'Numeric Variables']
    ) 

if nav == 'Home':
    if st.checkbox('<- For a first scan of the cars data, click here'):
        buffer = StringIO()
        cars.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    multi = '''We see that only one variable is categorical/nominal ("model"),  
    all other ones are numeric data. Of these latter, `mpg`, `disp`, `drat`, `wt`  
    and `qsec` are continuous. The rest are integer variables.'''
    st.markdown(multi)
    
if nav == 'Numeric Variables':
    st.write('Create plots for inspecting numeric variables.')
    if st.checkbox('<- Click here for inspecting summary stats'):
        st.table(cars.describe())
    
    multi = '''We could make first hypotheses about the distribution of our data.  
    * First working hypothesis:
        - numeric variables are normally distributed
    '''
    st.markdown(multi)

    if st.checkbox("<- Let's check box plots:"):
        boxplots = car_viz.boxplot(columns=numeric, y_max=None)
        try:
            st.pyplot(boxplots)
        except Exception as e:
            st.error('An error occurred: {}'.format(e))

    if st.checkbox('<- Click here for checking pairplots and scatterplots'):
        pairplot = car_viz.pairplot()
        try:
            st.pyplot(pairplot())
        except Exception as e:
            st.error('An error occurred: {}'.format(e))

    if st.checkbox('<- Click here for checking QQplots (assumption: vars are normally distributed)'):
        qqplot = car_viz.qq_plot()
        try:
            st.pyplot(qqplot())
        except Exception as e:
            st.error('An error occurred: {}'.format(e))