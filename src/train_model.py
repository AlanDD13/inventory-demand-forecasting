from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pickle
import pandas as pd
import os


def split_data(data, features, target, test_size=0.2):
    """
    Splits the data into training and testing datasets.

    Parameters:
    data (DataFrame): The input data.
    features (list): List of feature column names.
    target (str): The target variable name.
    test_size (float): Proportion of the data to be used for testing.

    Returns:
    X_train, X_test, y_train, y_test, train_data, test_data: Split data into training and testing sets.
    """
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.set_index('date', inplace=True)

    train_data = data[data.index < pd.Timestamp('2017-01-01')]
    test_data = data[data.index >= pd.Timestamp('2017-01-01')]

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    return X_train, X_test, y_train, y_test, train_data, test_data


def train_model(X_train, y_train, best_params=None):
    """
    Trains a Random Forest model with the provided parameters.

    Parameters:
    X_train (DataFrame): Feature data for training.
    y_train (Series): Target variable for training.
    best_params (dict, optional): Hyperparameters for the model. If None, default parameters are used.

    Returns:
    model: The trained Random Forest model.
    """
    if best_params is None:
        best_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 2,
            'random_state': 42
        }

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of the model using MAE and RMSE metrics.

    Parameters:
    model: The trained model.
    X_test (DataFrame): Feature data for testing.
    y_test (Series): True values for testing.

    Returns:
    dict: Dictionary containing MAE and RMSE metrics.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {"MAE": mae, "RMSE": rmse}


def save_model(model, filename='model.pkl'):
    """
    Saves the trained model to a file.

    Parameters:
    model: The trained model.
    filename (str, optional): The file name to save the model. Defaults to 'model.pkl'.
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_directory, filename)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Loads a trained model from a file.

    Parameters:
    filename (str): The name of the file containing the model.

    Returns:
    model: The loaded model.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    print(f"Model loaded from {filename}")
    return model
