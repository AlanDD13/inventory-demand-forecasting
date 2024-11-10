import matplotlib.pyplot as plt


def plot_predictions(test_data, y_true, y_pred):
    """
    Plots a comparison of actual vs. predicted values.

    Parameters:
    test_data (DataFrame): Test data containing indices for plotting.
    y_true (Series or array-like): The actual target values.
    y_pred (Series or array-like): The predicted target values.
    """
    # Configure plotting settings for handling large data sets
    plt.rcParams['agg.path.chunksize'] = 19000

    # Create the plot with a specified figure size
    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(test_data.index, y_true, label='Actual Sales', color='b', alpha=0.7, linewidth=2)

    # Plot predicted values
    plt.plot(test_data.index, y_pred, label='Predicted Sales (RF)',
             color='r', alpha=0.7, linestyle='--', linewidth=2)

    # Add axis labels and title
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Actual vs. Predicted Sales Comparison')

    # Add a legend for clarity
    plt.legend()

    # Display the plot
    plt.show()
