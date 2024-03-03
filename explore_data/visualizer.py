import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.figure_factory as ff

class Visualizer:
    '''Visualize different aspects of data.'''
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
    
    def evolution_over_time(self, column: pd.Series, y_max=None, **kwargs) -> plt.Axes:
        '''
        Visualize the evolution over time of a column.
        '''
        ax = self.data.plot.line(y=column, **kwargs)
        if y_max is not None:
            ax.set_ylim(0, y_max)
        return ax
    
    def boxplot(self, columns=None, y_max=None, figsize=(7, 5), **kwargs) -> plt.Figure:
        '''
        Generate box plots for selected columns.
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
        return fig

    def pairplot(self, y_max=None, **kwargs) -> plt.Figure:
        '''
        Generate a seaborn pairplot of vars included in the initial dataset.
        @Params:
            - y_max: Maximum value for the y-axis (optional).
            - kwargs: Keyword args to pass down to `sns.pairplot()`.
        @Return:
            - A matplotlib Figure object containing the pair plot.
        '''
        g = sns.pairplot(self.data, **kwargs)
        if y_max is not None:
            g.map(plt.ylim, (0, y_max))
        return g.fig
    
    def correlation_heatmap(self, y_max=None, **kwargs) -> plt.Axes:
        '''
        Plot the correlations between columns with a heatmap.
        '''
        corr = self.data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(8, 6), alpha=0.5)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=y_max if y_max is not None else 0.3, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            **kwargs
        )
        plt.title('Correlation')
        plt.tight_layout()
        return f

    def qq_plot(self, column_name: str, **kwargs) -> plt.Figure:
        '''
        Generate a Quantile-Quantile (Q-Q) plot for a selected column.
        @Params:
            - column_name: The name of the column to visualize.
            - kwargs: Additional keyword args to pass down to plotting function.
        '''
        try:
            column = self.data[column_name]
            fig, ax = plt.subplots()
            sm.qqplot(column, line='s', ax=ax, **kwargs)
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.set_title(f'Q-Q Plot for {column_name}')
            plt.close(fig)  # Close the figure to prevent it from being displayed twice
            return fig
        except KeyError:
            print(f"Error: '{column_name}' is not a valid column name in the dataset.")
            return None

    def distribution_col(self, column_name: str, bin_size: int = 10, y_max=None, **kwargs) -> plt.Figure:
        '''
        Generate a seaborn displot for a selected column.
        @Params:
            - column_name: The name of the column to visualize.
            - bin_size: The size of bins for histogram.
            - y_max: Maximum value for the y-axis (optional).
            - kwargs: Additional keyword args to pass down to sns.displot().
        '''
        try:
            column = self.data[column_name]
            fig = plt.figure(figsize=(8, 6))
            sns.histplot(column, bins=bin_size, **kwargs)
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.grid(True)
            if y_max is not None:
                plt.ylim(0, y_max)
            plt.tight_layout()
            return fig
        except KeyError:
            print(f"Error: '{column_name}' is not a valid column name in the dataset.")
            return None
        except TypeError:
            print("Error: Please provide a valid column name.")
            return None

    def distribution_grid(self, y_max=None, **kwargs) -> plt.Axes:
        '''
        Generate seaborn displots for all numeric columns.
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