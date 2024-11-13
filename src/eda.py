import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    """
    Loads data from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame: The loaded data.
    """
    data = pd.read_csv(filepath)
    return data


def initial_analysis(data):
    """
    Performs initial data analysis: displays basic information and descriptive statistics.

    Parameters:
    data (DataFrame): The input data.

    This function also plots the distribution of 'sales' column using a histogram.
    """
    print(data.info())
    print(data.describe())

    # Plotting distribution of sales
    sns.histplot(data['sales'], bins=30, kde=True)
    plt.title('Demand Distribution')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.show()
