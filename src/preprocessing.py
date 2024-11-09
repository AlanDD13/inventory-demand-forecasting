import pandas as pd
import numpy as np


def add_date_features(data, date_column='date'):
    """
    Добавляет признаки из даты, такие как месяц, день недели и год.
    
    Parameters:
    data (DataFrame): Данные.
    date_column (str): Название колонки с датой.
    
    Returns:
    DataFrame: Данные с добавленными признаками.
    """
    data[date_column] = pd.to_datetime(data[date_column])
    data['month'] = data[date_column].dt.month
    data['day_of_week'] = data[date_column].dt.dayofweek
    data['year'] = data[date_column].dt.year

    data.fillna(data.mean(), inplace=True)
    data.dropna()

    return data
