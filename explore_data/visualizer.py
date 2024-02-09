#visualizer.py
#TODO: implement boxplots per group (minority vs majority)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (roc_curve, auc)

import statsmodels.api as sm
#import pymc3 as pm
import scipy.stats as stats
#from pandas.plotting import _matplotlib

class UstVisualizer:
    '''Visualize different aspects of ust data.'''
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

        # Calculate the number of rows and columns for the facet grid
        num_cols = 3  # You can adjust the number of columns as needed
        num_rows = int(np.ceil(len(numeric.columns) / num_cols))

        # Create subplots with the specified figsize
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

        # Flatten the axes array
        axes = axes.flatten()

        # Iterate over the columns and plot each one individually
        for i, ax in enumerate(axes):
            if i < len(numeric.columns):
                col_name = numeric.columns[i]
                sns.boxplot(data=numeric[col_name], ax=ax, **kwargs)
                ax.set_title(col_name)
                ax.set_xlabel('')  # Remove x-label for clarity
                if y_max is not None:
                    ax.set_ylim(0, y_max)

        # Remove any unused axes
        for ax in axes[len(numeric.columns):]:
            ax.remove()

        plt.tight_layout()
        #return fig

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
    
    def qq_plot(self, column: pd.Series, **kwargs):
        '''
        Generate a Quantile-Quantile (Q-Q) plot for a selected column.
        @Params:
            - column: The column to visualize.
            - kwargs: Additional keyword args to pass down to plotting function.
        '''
        # Create a Q-Q plot using statsmodels
        sm.qqplot(column, line='s', **kwargs)
        
        # Add labels and title
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.title(f'Q-Q Plot for {column.name}')
        
        plt.show()

    def qq_plot_custom_distribution(self, data_column: pd.Series, pymc3_distribution, **kwargs):
        '''
        Generate a Q-Q plot for a specified PyMC3 distribution.
        @Params:
            - data_column: The column of your data to visualize.
            - pymc3_distribution: A PyMC3 distribution object representing the chosen distribution.
            - kwargs: Additional keyword args to pass to the plt.scatter function.
        '''
        with pm.Model() as model:
            # Create a custom random variable using the PyMC3 distribution
            rv = pymc3_distribution('rv', **kwargs)
            likelihood = pm.Normal('likelihood', mu=rv, sd=1, observed=data_column)
        
        with model:
            trace = pm.sample(1000, tune=1000)
        
        # Extract posterior samples
        samples = trace['rv']
        
        # Generate Q-Q plot
        stats.probplot(samples, dist=stats.norm, plot=plt)
        plt.title(f'Q-Q Plot for {pymc3_distribution.__name__} Distribution')
        plt.show()

    def plot_roc_curves(self, X_val: pd.DataFrame, y_val: pd.Series, classifiers: dict, target_names: list) -> None:
        '''Plot ROC curves for multiple classifiers.
        @Params:
            - X_val: validation set.
            - y_val: labels of validation set.
            - classifiers: a dictionary of classifiers as {classifier_name: classifier_instance}.
            - target_names: list of target class names.
        '''
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted. Call 'fit_model' first.")

        fig, ax = plt.subplots()

        for classifier_name, classifier_instance in classifiers.items():
            fpr, tpr, _ = roc_curve(y_val, classifier_instance.predict_proba(X_val)[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{classifier_name} (AUC = {roc_auc:.2f})', alpha=0.8)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Show the plot
        plt.show()
    
