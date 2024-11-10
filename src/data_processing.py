"""
Demand Forecasting Pipeline

This module processes data, performs feature engineering, splits data, trains a machine learning model, 
evaluates it, and visualizes predictions for demand forecasting.

Dependencies:
    - load_data (from src.data_loading)
    - add_date_features (from src.preprocessing)
    - plot_time_series (from src.eda)
    - split_data, train_model, evaluate_model, save_model (from src.train_model)
    - plot_predictions (from src.evaluate)
"""

from src.data_loading import load_data
from src.preprocessing import add_date_features
from src.eda import plot_time_series
from src.train_model import split_data, train_model, evaluate_model, save_model
from src.evaluate import plot_predictions


def data_processing(data_path='data/Store Demand Forecasting Train Data.csv'):
    """
    Main function to execute the demand forecasting data processing pipeline.

    Steps:
        1. Load the data from a specified path.
        2. Add date-related features to the data.
        3. Perform exploratory data analysis with a time series plot.
        4. Split data into training and testing sets.
        5. Train a machine learning model using the training data.
        6. Save the trained model.
        7. Evaluate the model using testing data.
        8. Visualize model predictions.

    Args:
        data_path (str): The file path to the input data. Default is 'data/Store Demand Forecasting Train Data.csv'.
    """
    # Load data
    data = load_data(data_path)

    # Add date-related features
    data = add_date_features(data, date_column='date')

    # Plot time series data for exploratory analysis
    plot_time_series(data, date_column='date', value_column='sales')

    # Define features and target variable
    features = ['item', 'store', 'month', 'day_of_week', 'year', 'day', 'week_of_year']
    target = 'sales'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, train_data, test_data = split_data(data, features, target)

    # Train the model using the training data
    model = train_model(X_train, y_train)

    # Save the trained model to a file
    save_model(model, '../model.pkl')

    # Evaluate the model using the testing data and print the metrics
    metrics = evaluate_model(model, X_test, y_test)
    print(f'Model Evaluation Metrics: {metrics}')

    # Predict on the test set and visualize the predictions
    predictions = model.predict(X_test)
    plot_predictions(test_data, y_test, predictions)


if __name__ == '__main__':
    data_processing()
