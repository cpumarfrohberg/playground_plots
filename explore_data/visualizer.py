#visualizer.py
#TODO: implement boxplots per group (minority vs majority)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
#import pymc3 as pm
import scipy.stats as stats
#from pandas.plotting import _matplotlib

class Visualizer:
    '''Visualize different aspects of  data.'''
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
    
    def evolution_over_time(self, column: pd.Series, y_max=None, **kwargs) -> plt.Axes:
        '''
        Visualize the evolution over time of a col.
        @Params:
            - column: Name of col to visualize.
            - y_max: Maximum value for the y-axis (optional).
            - kwargs: Additional keyword args to pass down to
                        plotting function.
        '''
        ax = self.data.plot.line(y=column, **kwargs)
        if y_max is not None:
            ax.set_ylim(0, y_max)
        return ax
    
    def boxplot(self, columns=None, y_max=None, figsize=(7, 5), **kwargs) -> sns.FacetGrid:
        '''
        Generate box plots for selected columns in a facet grid.
        @Params:
            - columns: List of column names to visualize (optional, defaults to all numeric columns).
            - y_max: Maximum value for the y-axis (optional).
            - figsize: Tuple specifying the figure size (width, height).
            - kwargs: Additional keyword args to pass down to `sns.boxplot()`.
        @Return:
            - A seaborn FacetGrid.
        '''
        if columns is None:
            numeric = self.data.select_dtypes(include='number')
        else:
            numeric = self.data[columns]

        num_cols = 3
        num_rows = int(np.ceil(len(numeric.columns) / num_cols))

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < len(numeric.columns):
                col_name = numeric.columns[i]
                sns.boxplot(data=numeric[col_name], ax=ax, **kwargs)
                ax.set_title(col_name)
                ax.set_xlabel('')  # Remove x-label for clarity
                if y_max is not None:
                    ax.set_ylim(0, y_max)
                    
        for ax in axes[len(numeric.columns):]:
            ax.remove()

        plt.tight_layout()

    def pairplot(self, y_max=None, **kwargs) -> sns.PairGrid:
        '''
        Generate a seaborn pairplot of vars included in the initial dataset.
        @Params:
            - y_max: Maximum value for the y-axis (optional).
            - kwargs: Keyword args to pass down to `sns.pairplot()`.
        @Return:
            - A seaborn pairplot.
        '''
        g = sns.pairplot(self.data, **kwargs)
        if y_max is not None:
            g.map(plt.ylim, (0, y_max))
        return g
    
    def correlation_heatmap(self, y_max=None, **kwargs) -> plt.Axes:
        '''
        Plot the correlations between columns with a heatmap.
        @Params:
            - y_max: Maximum value for the y-axis (optional).
            - kwargs: Keyword args to pass down to `sns.heatmap()`.
        '''
        corr = self.data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(8, 6), alpha=0.5)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=y_max if y_max is not None else 0.3,  # Set the maximum value for the color scale 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            **kwargs
        )
        plt.title('Correlation')
        plt.tight_layout()
        plt.show()

    def qq_plot(self, column: pd.Series, **kwargs):
        '''
        Generate a Quantile-Quantile (Q-Q) plot for a selected column.
        @Params:
            - column: The column to visualize.
            - kwargs: Additional keyword args to pass down to plotting function.
        '''
        sm.qqplot(column, line='s', **kwargs)
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.title(f'Q-Q Plot for {column.name}')
        plt.show()

    def distribution_col(self, column: pd.Series, y_max=None, **kwargs) -> sns.displot:
        '''
        Generate a seaborn displot for a selected column.
        @Params:
            - column: The column to visualize.
            - y_max: Maximum value for the y-axis (optional).
            - kwargs: Keyword args to pass down to `sns.displot()`.
        '''
        g = sns.displot(
            data=self.data,
            x=column,
            kde=True,
            aspect=2.5,
            height=3.5,
            alpha=0.5,
            **kwargs
        )
        if y_max is not None:
            g.set(ylim=(0, y_max))
        plt.title(f'Distribution of {column.name}')
        plt.tight_layout()
        plt.show()

    def distribution_grid(self, y_max=None, **kwargs) -> plt.Axes:
        '''
        Generate seaborn displots for all numeric columns.
        @Params:
            - y_max: Maximum value for the y-axis (optional).
            - kwargs: Keyword args to pass down to `sns.displot()`.
        '''
        numeric = self.data.select_dtypes(include='number')
        g = numeric.hist(
            figsize=(15, 10),
            alpha=0.5,
            **kwargs
        )
        if y_max is not None:
            for ax in g.flatten():
                ax.set_ylim(0, y_max)
        return g
    
    # def qq_plot_custom_distribution(self, data_column: pd.Series, pymc3_distribution, **kwargs):
    #     '''
    #     Generate a Q-Q plot for a specified PyMC3 distribution.
    #     @Params:
    #         - data_column: The column of your data to visualize.
    #         - pymc3_distribution: A PyMC3 distribution object representing the chosen distribution.
    #         - kwargs: Additional keyword args to pass to the plt.scatter function.
    #     '''
    #     with pm.Model() as model:
    #         # Create a custom random variable using the PyMC3 distribution
    #         rv = pymc3_distribution('rv', **kwargs)
    #         likelihood = pm.Normal('likelihood', mu=rv, sd=1, observed=data_column)
        
    #     with model:
    #         trace = pm.sample(1000, tune=1000)
        
    #     # Extract posterior samples
    #     samples = trace['rv']
        
    #     # Generate Q-Q plot
    #     stats.probplot(samples, dist=stats.norm, plot=plt)
    #     plt.title(f'Q-Q Plot for {pymc3_distribution.__name__} Distribution')
    #     plt.show()
    
