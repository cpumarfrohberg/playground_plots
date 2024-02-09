#transformer.py
#TODO: include **frequency** of transactions & counts of changes in score per client (in time interval of interest)


import pandas as pd
import datetime as dt

class rfmTransformer:
    '''Return initial DataFrame with additional cols created based on RFM.
    "Monetary": variable "amount" in initial dataset.
    "Recency": timedelta between specified date and initial date of validity of score.
    "Frequency": track changes/updates of scores within prespecified time interval.

    @Params:
        - data: dataframe from which to create RFM-features.
        - y_max: Maximum value for the y-axis (optional).
        - kwargs: Additional keyword args to pass down to
                    plotting function.
    @Return:
        - data: initial pd.DataFrame with additional columns, `recency` and `frequency`.
    '''

    def make_RFM_features(self, data: pd.DataFrame)->pd.DataFrame:
    
        # Calculate Recency (R)
        current_date = dt.datetime(2023, 9, 8)
        data['recency'] = (current_date - data['valid_from']).dt.days

        # Calculate Frequency (F)
        time_frame = dt.timedelta(days=90)
        data['frequency'] = data.apply(
            lambda row: data[
                (data['valid_from'] >= row['valid_from'] - time_frame) &
                (data['valid_from'] <= row['valid_from'])].shape[0],
                axis=1
                ) 

        data_sorted = data.sort_values(by='valid_from')

        frequency_values = list()

        for index, row in data_sorted.iterrows():
            current_date = row['valid_from']
            
            # Calculate relevant time interval
            start_date = current_date - time_frame
            end_date = current_date
            
            # Filter the DataFrame to get updates within the interval
            updates_within_time_frame = data_sorted[
                (data_sorted['valid_from'] >= start_date) &
                (data_sorted['valid_from'] <= end_date)
                ]
            
            frequency = len(updates_within_time_frame) # Count the number of updates
            frequency_values.append(frequency)

        data['frequency'] = frequency_values
        return data