# Импорт всех необходимых модулей
from src.data_loading import load_data, initial_analysis
from src.preprocessing import add_date_features
from src.eda import plot_time_series
from src.train_model import split_data, train_model, evaluate_model, save_model
from src.evaluate import plot_predictions


def data_processing(data_path='data/Store Demand Forecasting Train Data.csv'):
    
    # 1. Загрузка данных и начальный анализ
    data = load_data(data_path)
    # initial_analysis(data)
    
    # 2. Предобработка данных
    data = add_date_features(data, date_column='date')
    
    # 3. Визуализация данных
    # plot_time_series(data, date_column='date', value_column='sales')
    
    # 4. Определение признаков и целевой переменной
    features = ['item', 'store', 'month', 'day_of_week', 'year']  # Добавь нужные признаки на основе своих данных
    target = 'sales'
    
    # 5. Разделение данных на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test, train_data, test_data = split_data(data, features, target)

    # 6. Обучение модели
    model = train_model(X_train, y_train)

    # 7. Сохранение модели
    save_model(model, '../model.pkl')

    # 8. Оценка модели
    # metrics = evaluate_model(model, X_test, y_test)
    # print(f'Model Evaluation Metrics: {metrics}')

    # 9. Визуализация прогнозов
    # predictions = model.predict(X_test)
    # plot_predictions(test_data, y_test, predictions)


if __name__ == '__main__':
    data_processing()

