#reader.py
"""Read, rename and reencode cols of boniversum data."""
import os
import pandas as pd
import numpy as np

PATH = '~/data/maynas'

class Reader:
    '''Read, rename and drop cols.'''

    def __init__(self, path = PATH):
        self.path = path

    def read_data(self, file: str) -> pd.DataFrame: #TODO: include drop cols option
        '''Return DataFrame with feature matrix and labels as values.'''
        df = pd.read_csv(f'{self.path}/{file}.csv', index_col=0, parse_dates=True, encoding = 'unicode_escape')
        return df

    def include_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Return DataFrame with time-stamps.'''
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['week_day'] = df.index.weekday
        df['week'] = df.index.week
        df['year'] = df.index.year
        df['month'] = df.index.month
        return df

    def time_parser(self, df: pd.DataFrame, time_parsables: list) -> pd.DataFrame:
        '''Parse columns encoded as strings to datetime-objects.'''
        for col in time_parsables:
            df[col] = pd.to_datetime(df[col])
        return df
    
    def save_intermediate_data(self, df: pd.DataFrame, file: str):
        '''Save extracted data locally as csv-file.'''
        return df.to_csv(f'{self.path}/{file}.csv')
