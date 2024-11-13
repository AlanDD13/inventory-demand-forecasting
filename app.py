"""
Inventory Demand Forecasting Application

This module is used for visualizing and predicting demand using machine learning models. 
It leverages data preprocessing, EDA (Exploratory Data Analysis), and prediction visualization features.

Dependencies:
    - pandas
    - streamlit
    - matplotlib
    - seaborn
    - pickle
    - os
"""

import datetime
import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import add_date_features
from src.model_building import data_processing


def load_model():
    """
    Loads the pre-trained model from the specified pickle file.

    Returns:
        model: The loaded model object or None if the model creation fails.
    """
    model_path = 'model.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)

    st.error("Model not found. Creating model...")
    success = data_processing()
    if success and os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error("Model creation failed. Please check the data processing pipeline.")
        return None


def main():
    """
    Main function to render the Streamlit web app for inventory demand forecasting.
    It displays an introduction, EDA, feature engineering steps, model prediction, 
    and visualization of the forecast results.
    """
    st.set_page_config(page_title="Inventory Demand Forecasting Presentation")

    st.title("Введение в проект")
    st.write("""
        ### Цель:

        Предсказать будущий спрос на товары в рамках логистической системы компании, 
        чтобы оптимизировать запасы и минимизировать расходы. 
        Прогнозирование спроса помогает компаниям быть готовыми к изменениям на рынке и улучшать процессы поставки.

        ### Методология:

        В рамках данного проекта используется метод машинного обучения, 
        а именно алгоритм Random Forest Regressor. 
        Я выбрал эту модель, 
        потому что она хорошо справляется с задачами регрессии и справляется с нелинейными зависимостями в данных.

        ### Основные этапы проекта:
        1. Сбор и подготовка данных.
        2. Проведение разведочного анализа данных (EDA).
        3. Обработка признаков и создание модели.
        4. Прогнозирование и анализ результатов.
    """)

    # Load and display data
    data = pd.read_csv('data/Store Demand Forecasting Train Data.csv')
    data = add_date_features(data)
    st.write("Пример данных:")
    st.dataframe(data)

    # EDA Section
    st.title("Разведочный анализ данных (EDA)")
    st.write("На данном этапе мы проверяем данные на наличие пропусков, аномалий, исследуем закономерности.")

    st.write("### Основные статистики")
    st.write(data.describe())

    st.write("### Распределение целевой переменной")
    fig, ax = plt.subplots()
    sns.histplot(data['sales'], kde=True, ax=ax)
    ax.set_title('Распределение продаж')
    ax.set_xlabel('Продажи')
    ax.set_ylabel('Частота')
    st.pyplot(fig)

    st.write("### Корреляции")
    corr_matrix = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    st.pyplot(fig)

    # Monthly Demand Visualization
    st.write("### Спрос за месяца")
    monthly_demand = data.groupby('month')['sales'].sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_demand.plot(kind='bar', ax=ax, title='Спрос за месяца', color='skyblue')
    ax.set_xlabel('Месяц')
    ax.set_ylabel('Спрос')
    st.pyplot(fig)

    # Weekly Demand Visualization
    st.write("### Средний спрос по дням недели")
    weekly_demand = data.groupby('day_of_week')['sales'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    weekly_demand.plot(kind='bar', ax=ax, title='Средний спрос по дням недели', color='skyblue')
    ax.set_xlabel('День недели')
    ax.set_ylabel('Средний спрос')
    ax.set_xticklabels(['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье'])
    st.pyplot(fig)

    # Demand Distribution by Store
    st.write("### График распределения спроса:")
    fig, ax = plt.subplots()
    sns.barplot(x='store', y='sales', data=data, ax=ax, color='skyblue')
    ax.set_title("Распределение спроса по магазинам")
    ax.set_xlabel('Магазин')
    ax.set_ylabel('Спрос')
    st.pyplot(fig)

    st.title("Обработка признаков")
    st.write("""            
            1.	Обработка данных:
            На этом этапе я очистил и преобразовал данные. 
            Были удалены пропущенные значения, а также проведена нормализация и стандартизация числовых признаков, 
            чтобы улучшить работу модели.

            2.	Создание новых признаков:
            Важным шагом стало создание дополнительных признаков (feature engineering).
            Я использовал различные метки времени, такие как день недели, месяц, неделя года,
            чтобы учесть сезонность и цикличность спроса.
    """)

    st.title("Создание модели и результаты")
    st.write("""
            3.	Обучение и проверка модели:
            Модель была обучена на данных, и для оценки ее качества использовались метрики, 
            такие как Mean Absolute Error (MAE). 
            Я протестировал несколько моделей, чтобы выбрать оптимальную.
            Для прогнозирования использовалась модель **Random Forest Regressor**.
            Я разделил данные на обучающую и тестовую выборки по временному признаку чтобы избежать утечки данных и 
            для проверки качества модели.
    """)

    # Load Model and Predict
    model = load_model()
    if model is not None:
        st.success("Модель успешно загружена.")
        st.write("4. Пример предсказания на основе данных пользователя:")

        # User Input for Predictions
        store = st.slider("Магазин (Store ID)", min_value=1, max_value=10, value=1, step=1)
        item = st.slider("Товар (Item ID)", min_value=1, max_value=50, value=1, step=1)
        min_date = datetime.date(2013, 1, 1)
        max_date = datetime.date(2025, 12, 31)
        selected_date = st.date_input("Дата: (date)", min_value=min_date, max_value=max_date)

        if st.button('Прогнозировать'):
            data_custom = pd.DataFrame({
                'item': [item],
                'store': [store],
                'month': [int(selected_date.strftime('%m'))],
                'day_of_week': [int(selected_date.strftime('%u')) - 1],
                'year': [int(selected_date.strftime('%Y'))],
                'day': [int(selected_date.strftime('%d'))],
                'week_of_year': [int(selected_date.strftime('%U')) + 1]
            })
            st.dataframe(data_custom)
            prediction = model.predict(data_custom)
            st.write(f"Прогнозируемое количество товаров: **{prediction[0]:.2f}**")
            # Visualization of Predictions
            st.title("Визуализация прогнозов")
            selected_rows = data[(data['month'] == int(selected_date.strftime('%m')))
                                 & (data['store'] == store) & (data['item'] == item)
                                 & (data['day'] == int(selected_date.strftime('%d')))
                                 & (data['year'] == int(selected_date.strftime('%Y')))]
            st.dataframe(selected_rows)
            fig, ax = plt.subplots()
            ax.bar(["Прогноз"], [prediction[0]], color="skyblue")
            ax.bar(['Действительность'], selected_rows['sales'], color="red")
            ax.set_ylabel('Количество товаров')
            ax.set_title("Прогноз спроса")
            st.pyplot(fig)

    st.title("Заключение")
    st.write("""
        5.	Модульность и воспроизводимость:
        Для удобства и воспроизводимости была организована структура кода с использованием Jupyter Notebooks 
        и интеграцией с Docker. Это позволяет легко запускать и повторять эксперименты в разных средах

        Проект по прогнозированию спроса на товары помогает компаниям улучшить управление запасами 
        и уменьшить издержки, эффективно реагируя на изменения рыночного спроса.

        В этом проекте я показал:
        - Использование данных для предсказания спроса.
        - Применение методов машинного обучения для улучшения точности прогноза.
        - Потенциальные улучшения для дальнейшего исследования: добавление новых признаков, 
        использование других моделей.
""")
    st.write("Спасибо за внимание!")


if __name__ == "__main__":
    main()
