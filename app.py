from explore_data.visualizer import Visualizer
from explore_data.reader import Reader
import streamlit as st
from io import StringIO
import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm



url = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'

@st.cache_data
def load_data():
    '''Read raw data and return DataFrame(s).'''
    reader = Reader()
    cars = reader.read_data(url)
    return cars

def instantiate_visualizer(data):
    '''Instantiate Visualizer object.'''
    return Visualizer(data)

cars = load_data()
car_viz = instantiate_visualizer(cars)
numeric = cars.select_dtypes(include=['int64', 'float64'])
car_viz_numeric = instantiate_visualizer(numeric)

st.title('Example Visual Analysis')

nav = st.sidebar.radio(
    'Please choose one of the following:',
    ['Home', 'Numeric Variables', 'Categorical Variables']
    ) 

if nav == 'Home':
    st.markdown(
    '''#### This is an example of how to visualize tabular data and make initial inferences.'''
    )
    if st.checkbox('<- For a first scan of the cars data, click here'):
        buffer = StringIO()
        cars.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    multi = '''
    * We see that 
        - only one variable is categorical/nominal (`model`), all other ones are numeric data
        - of the latter, `mpg`, `disp`, `drat`, `wt`, and `qsec` are continuous. The rest are integer variables.
        - there's no missing data
        - sample size is relatively small (additional challenge for running significance tests)
    '''
    st.markdown(multi)
    
if nav == 'Numeric Variables':
    st.write('Create plots for inspecting numeric variables.')
    if st.checkbox('<- Click here for inspecting summary stats'):
        st.table(cars.describe())
    
        first_working_hypo = '''
        * First working hypothesis: **variables are normally distributed**. 
            - however, when comparing mean and median values in the summary stats, we note that this might not be the case (e.g., `disp`, but also `hp`...and other ones - can you spot which ones?)
            - in fact, by comparing named values with each other, we expect that either the data-generating process (DGP) is non-normal or that we have outliers.
            - in addition, and by comparing the first and third quartiles with the median, we can check if our variables are left (`disp`) or right-skewed (e.g., `hp`)
            - **we will use plots to see if our hypotheses hold**
        '''
        st.markdown(first_working_hypo)
        
        second_working_hypo = '''
        * Second working hypothesis: **some variables are correlated**:
            - we would expect some variables to be correlated one with another, e.g. horsepower (`hp`) and mileage per gallon (`mpg`) (expectation: positive corr)
        * **we will use plots to see if our hypotheses hold**
        '''
        st.markdown(second_working_hypo)

    if st.checkbox("<- Click here in order to check box plots"):
        boxplots = car_viz.boxplot(columns=numeric.columns, y_max=None)
        st.pyplot(boxplots)

        interpret_boxplots = '''
        * Re first working hypothesis: 
            - **only *one* variable seems to be normally distributed: `cyl`**
            - also, all other variables are either left or right-skewed
            - additionally, we see that for `mpg`, `hp`, `qsec`, `carb` and `wt`, we either might have outliers or we are looking into a non-normal DGP
                - this indicates we should expand our analysis to an analysis of correlation (since we could simply have non-normal distributions and "outliers" actually are not outliers, but simply follow a non-normal distribution)
            - we will **need to use additional plots and/or run statistical tests in order to see if our hypotheses actually hold**
        '''
        st.markdown(interpret_boxplots)

    if st.checkbox('<- Click here for checking pair plots'):
        pairplot = car_viz.pairplot()
        st.write(pairplot)
    
        interpret_pairplot = '''
        * Re first working hypothesis: **by inspecting the histograms of the numeric data, we cannot be sure that **any of the variables** are normally distributed**
            - by looking at the histograms, we notice that
                - histograms are not helpful for `int64` column types produce distorted histograms given the duplicated counts of individual vals (e.g. for `cyl` variable)
                - histograms for `float64` column types don't resemble a normal distribution; this is true also for different bin sizes (convince yourself of this below in next section)
            - in addition, we can see that some variables seem to be correlated one with the other, mainly negatively so (!)
            - **we will need to user additional plots and/or run statistical tests in order to see if our second hypothesis holds**
        '''
        st.markdown(interpret_pairplot)

        with st.expander('Investigate here how histograms of continous data changes for varying bin sizes:'):
            column_name = st.selectbox('Select a column:', numeric.columns[numeric.dtypes == 'float64'])
            bin_size = st.slider('Select bin size:', min_value=1, max_value=50, value=10)
            distribution_continuous = car_viz.distribution_col(column_name, bin_size=bin_size)
            st.pyplot(distribution_continuous)

            interpret_distribution_plots = '''
            * we can conclude that even after varying bin sizes, continous data does not seem to follow a normal distribution.
            '''
            st.markdown(interpret_distribution_plots)

    if st.checkbox('<- Click here for checking QQ plots'):
        column_name = st.selectbox('Select a column:', numeric.columns.tolist())
        qqplot = car_viz.qq_plot(column_name)
        st.pyplot(qqplot)

        interpret_qqplot = '''
        * Re first working hypothesis: **we cannot be sure that any of the variables are normally distributed**
            - rather, we see that only `mpg` seems to comply to this assumption (since most of the datapoints lie on the line of of the theoretical quantiles)
            - at the same time, the datapoints which are relatively far from the line indicate either the existence of tails or of outliers
            - finally, we see that the integer variables (e.g. `cyl`) require additional tests in order to check for normality (horizontal lines are due to duplicated values, which result from identical decimal values)
            - we will need to **use additional plots and/or run statistical tests in order to see if our second  
            hypothesis holds**
        '''
        st.markdown(interpret_qqplot)
    
    if st.checkbox('<- Click here for plotting correlation heat map'):
        corr_heatmap = car_viz_numeric.correlation_heatmap()
        st.pyplot(corr_heatmap)

        interpret_corr_heatmap = '''
        * Re corrrelation: relatively strong negative correlation between selected vars, while positive correlations rather weak
            - in addition: as we expected, `hp` and `mpg` are correlated with each other but **negatively** so (`rho` ~ -0.8 )
                - in fact `mpg` is the variable which correlates with the greatest amount of other vars; this could be a hint of multicolinearity, which would imply thinking about  
                what to do with that var in the context of machine learning (e.g. linear regression with `mpg` being the dependent variable)
            - also, and with regards to weaker correlations between the other vars: we should run additional tests in order to test for statistical significance
        '''
        st.markdown(interpret_corr_heatmap)