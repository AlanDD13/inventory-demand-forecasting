from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pickle
import pandas as pd


def split_data(data, features, target, test_size=0.2):
    """
    Разделяет данные на тренировочные и тестовые выборки.
    
    Parameters:
    data (DataFrame): Исходные данные.
    features (list): Список признаков.
    target (str): Название целевой переменной.
    test_size (float): Доля тестовой выборки.
    
    Returns:
    X_train, X_test, y_train, y_test: Разделенные данные.
    """
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.set_index('date', inplace=True)
    train_data = data[data.index < pd.Timestamp('2017-01-01')]
    test_data = data[data.index >= pd.Timestamp('2017-01-01')]

    # Separating features and target variable
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    return X_train, X_test, y_train, y_test, train_data, test_data


def train_model(X_train, y_train, best_params=None):
    """
    Обучает модель Random Forest с оптимальными параметрами, если они заданы.
    
    Parameters:
    X_train (DataFrame): Признаки тренировочной выборки.
    y_train (Series): Целевая переменная.
    best_params (dict): Оптимальные параметры для модели.
    
    Returns:
    model: Обученная модель.
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
    Оценивает модель с использованием MAE и RMSE.
    
    Parameters:
    model: Обученная модель.
    X_test (DataFrame): Признаки тестовой выборки.
    y_test (Series): Целевая переменная.
    
    Returns:
    dict: Метрики качества.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {"MAE": mae, "RMSE": rmse}


def save_model(model, filename='model.pkl'):
    """
    Сохраняет модель в файл.
    
    Parameters:
    model: Обученная модель.
    filename (str): Имя файла для сохранения.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Загружает модель из файла.
    
    Parameters:
    filename (str): Имя файла для загрузки.
    
    Returns:
    model: Загруженная модель.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model
