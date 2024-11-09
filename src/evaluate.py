import matplotlib.pyplot as plt


def plot_predictions(test_data, y_true, y_pred):
    """
    Строит график сравнения истинных и предсказанных значений.
    
    Parameters:
    y_true (Series): Истинные значения.
    y_pred (Series): Предсказанные значения.
    """
    plt.rcParams['agg.path.chunksize'] = 19000  # You can increase this value further if needed

    # Optionally, reduce the complexity of paths by increasing the simplify threshold
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, y_true, label='Actual Sales', color='b')
    plt.plot(test_data.index, y_pred, label='Predicted Sales (RF)', color='r')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Actual vs. Predicted Sales')
    plt.legend()
    plt.show()
