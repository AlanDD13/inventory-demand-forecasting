import matplotlib.pyplot as plt


def plot_time_series(data, date_column='date', value_column='sales'):
    """
    Строит временной график для анализа изменения спроса.
    
    Parameters:
    data (DataFrame): Данные.
    date_column (str): Колонка с датой.
    value_column (str): Колонка с значениями для анализа.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(data[date_column], data[value_column])
    plt.title('Demand Over Time')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.show()
