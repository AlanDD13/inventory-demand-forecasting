import pandas as pd


def add_date_features(data, date_column='date'):
    """
    Adds date-related features such as month, day of the week, year, day,
    and week of the year from a specified date column.

    Parameters:
    data (DataFrame): The input DataFrame containing data.
    date_column (str): The name of the column containing date values.

    Returns:
    DataFrame: The DataFrame with added date-related features.
    """
    # Convert the specified date column to datetime format
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    # Extract date-related features
    data['month'] = data[date_column].dt.month
    data['day_of_week'] = data[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
    data['year'] = data[date_column].dt.year
    data['day'] = data[date_column].dt.day
    data['week_of_year'] = data[date_column].dt.isocalendar().week

    # Handle missing values by filling them with the mean for numerical columns
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Optional: drop remaining rows with missing values if any
    data.dropna(inplace=True)

    return data
