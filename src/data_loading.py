import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    """
    Загружает данные из CSV файла.
    
    Parameters:
    filepath (str): Путь к файлу.
    
    Returns:
    DataFrame: Загруженные данные.
    """
    data = pd.read_csv(filepath)
    return data


def initial_analysis(data):
    """
    Проводит начальный анализ данных: просмотр информации и описательной статистики.
    
    Parameters:
    data (DataFrame): Исходные данные.
    """
    print(data.info())
    print(data.describe())
    sns.histplot(data['sales'], bins=30, kde=True)
    plt.title('Demand Distribution')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.show()
    