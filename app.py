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

nav = st.sidebar.radio(
    'Please chose one of the following:',
    ['Home', 'Numeric Variables', 'Categorical Variables']
    ) 

if nav == 'Home':
    st.markdown(
    ''' #### This is an example on how to visualize tabular data and on how to make initial inferences.
    '''
    )
    if st.checkbox('<- For a first scan of the cars data, click here'):
        buffer = StringIO()
        cars.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    multi = '''We see that only one variable is categorical/nominal (`model`), all other ones are numeric data.  
    Of these latter, `mpg`, `disp`, `drat`, `wt` and `qsec` are continuous. The rest are integer variables.'''
    st.markdown(multi)
    
if nav == 'Numeric Variables':
    st.write('Create plots for inspecting numeric variables.')
    if st.checkbox('<- Click here for inspecting summary stats'):
        st.table(cars.describe())
    
    first_working_hypo = '''* First working hypothesis: **variables are normally distributed**. However:
    - comparing mean and median vals, we note that this might not be the case (e.g. `disp`,  
    but also `hp`...and other ones - can you spot which ones?)
    - in fact, by comparing mean and median vals, we can actually already say that either  
    the data-generating processes (DGP) are non-uniform or that we have outliers.
    - in addition, and by comparing the first and third quartiles with the median, we can  
    check if our vars are left (`disp`) or right-skewed (e.g. `hp`)
    - **we will use plots in order to see if our hypotheses could are true**
    '''
    st.markdown(first_working_hypo)
    
    second_working_hypo = '''* Second working hypothesis: **some variables are correlated**:
    - we would expect some variables to be correlated with each other, e.g. horse-power (i.e.  
    `hp` and mileage per gallon (`mpg`) to be positively correlated)
    - **we will use plots in order to see if our hypotheses could are true**
    '''
    st.markdown(second_working_hypo)

    if st.checkbox("<- Let's check box plots:"):
        boxplots = car_viz.boxplot(columns=numeric, y_max=None)
        try:
            st.pyplot(boxplots)
        except Exception as e:
            st.error('An error occurred: {}'.format(e))

    if st.checkbox('<- Click here for checking pairplots and scatterplots'):
        pairplot = car_viz.pairplot()
        try:
            st.write(pairplot())
        except Exception as e:
            st.error('An error occurred: {}'.format(e))

    if st.checkbox('<- Click here for checking QQplots (assumption: vars are normally distributed)'):
        qqplot = car_viz.qq_plot()
        try:
            st.write(qqplot())
        except Exception as e:
            st.error('An error occurred: {}'.format(e))