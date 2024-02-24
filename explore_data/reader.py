#reader.py
"""Read, rename and reencode cols of mtcars data."""
import os
import pandas as pd
import numpy as np

url = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'

class Reader:
    '''Read, rename and drop cols.'''

    def read_data(self, url: str) -> pd.DataFrame: #TODO: include drop cols option
        '''Return DataFrame with feature matrix and labels as values.'''
        df = pd.read_csv(url)
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
