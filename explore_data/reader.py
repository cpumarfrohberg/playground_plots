#reader.py
"""Read, rename and reencode cols of boniversum data."""
import os
import pandas as pd
import numpy as np
import snowflake

from sklearn.model_selection import GridSearchCV

#from explore_data.utils import PATH

PATH = '~/projects/data/score_providers/boniversum'

class BoniversumReader:
    '''Read, rename and drop cols.'''

    def __init__(self, path = PATH):
        self.path = path
    
    def get_data_from_snowflake(self, query:str)->pd.DataFrame:
        '''Extract internal data.'''
        con = snowflake.connector.connect(
        user=os.environ['DWH_USER_SNOWFLAKE'],
        password=os.environ['DWH_PASSWORD_SNOWFLAKE'],
        authenticator='externalbrowser',
        account=os.environ['DWH_ACCOUNT_SNOWFLAKE']
        )
        cur = con.cursor()
        try:
            cur.execute(query)

            names = [ x[0].lower() for x in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame( rows, columns=names)
        finally:
            cur.close()
        return df
    
    def write_data_artefacts(self, data_to_write:pd.DataFrame, data_artefact_name:str)->None:
        '''Write data artefacts.
        @Params:
            - data_to_write: name of pd.DataFrame to be saved
            - data_artefact_name: name of data artefact saveable
        @Returns:
            - None
        '''
        data_to_write.to_csv(f'{self.path}/{data_artefact_name}.csv')
    
    def write_best_params(self, grid_search_object:None, grid_search_artefact_name:str)->None:
        '''Write data artefacts.
        @Params:
            - data_to_write: name of pd.DataFrame to be saved
            - data_artefact_name: name of data artefact saveable
        @Returns:
            - None
        '''
        best_params = grid_search_object.best_params_
        best_params_df = pd.DataFrame([best_params]).T
        best_params_df.columns = ['Best Parameters']
        best_params_df.to_csv(f'{grid_search_artefact_name}.csv', index=True)

    def create_overview_scores(self, list_scores_tables: list)->pd.DataFrame:
        '''Return concatenated DataFrame including overview of performance scores.'''
        frames_concatenable = list()
        for score_summary in list_scores_tables:
            individual_summary = score_summary.iloc[0, 1:]
            individual_summary = individual_summary.to_frame()
            frames_concatenable.append(individual_summary)
        return frames_concatenable
