import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Функция для загрузки модели
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Файл с моделью не найден. Убедитесь, что 'model.pkl' доступен.")
        return None


# Основное приложение Streamlit
def main():
    st.set_page_config(page_title="Inventory Demand Forecasting Presentation", layout="wide")

    # Презентация в виде слайдов
    st.sidebar.title("Навигация по слайдам")
    slide = st.sidebar.radio("Выберите слайд:",
                             ["Введение", "Исследование данных", "Обработка признаков", "Модель и результаты",
                              "Визуализация прогнозов", "Заключение"])

    if slide == "Введение":
        st.title("Введение в проект")
        st.write("""
            Цель проекта - прогнозирование спроса на товары в магазинах на основе исторических данных.
            Это помогает оптимизировать управление запасами и сократить издержки.

            Основные этапы проекта:
            1. Сбор и подготовка данных.
            2. Проведение разведочного анализа данных (EDA).
            3. Обработка признаков и создание модели.
            4. Прогнозирование и анализ результатов.
        """)
        st.image("path_to_your_intro_image.png")  # Можно добавить изображение, если есть

    elif slide == "Исследование данных":
        st.title("Разведочный анализ данных (EDA)")
        st.write("На данном этапе мы проверяем данные на наличие пропусков, аномалий, исследуем закономерности.")

        # Пример демонстрации DataFrame (заглушка для данных)
        data = pd.DataFrame({
            'store': [1, 2, 3],
            'item': [1, 2, 3],
            'sales': [20, 30, 40],
            'month': [1, 2, 3],
            'day_of_week': [0, 1, 2],
            'year': [2013, 2014, 2015]
        })
        st.write("Пример данных:")
        st.dataframe(data)

        # Пример визуализации
        st.write("График распределения продаж:")
        fig, ax = plt.subplots()
        sns.barplot(x='store', y='sales', data=data, ax=ax)
        ax.set_title("Распределение продаж по магазинам")
        st.pyplot(fig)

    elif slide == "Обработка признаков":
        st.title("Обработка признаков")
        st.write("""
            Для улучшения качества модели выполняется обработка признаков:
            - Создание новых признаков (например, сезонные факторы).
            - Удаление нерелевантных данных.
            - Кодирование категориальных признаков.
        """)
        st.image("path_to_feature_processing_image.png")  # Пример изображения или визуализации

    elif slide == "Модель и результаты":
        st.title("Создание модели и результаты")
        st.write("""
            Для прогнозирования использовалась модель **Random Forest Regressor**.
            Мы разделили данные на обучающую и тестовую выборки для проверки качества модели.
        """)
        model = load_model()
        if model is not None:
            st.success("Модель успешно загружена.")
            st.write("Пример предсказания на основе данных пользователя:")

            # Заглушка ввода данных (для демонстрации)
            data = pd.DataFrame({
                'store': [1],
                'item': [1],
                'month': [1],
                'day_of_week': [0],
                'year': [2013]
            })
            prediction = model.predict(data)
            st.write(f"Прогнозируемое количество товаров: **{prediction[0]:.2f}**")

            st.title("Визуализация прогнозов")
            st.write("Результаты прогнозов отображаются с помощью графиков, что позволяет лучше понять динамику спроса.")

            # Пример визуализации прогноза
            fig, ax = plt.subplots()
            ax.bar(["Прогноз"], [prediction[0]], color="skyblue")
            ax.set_ylabel('Количество товаров')
            ax.set_title("Прогноз спроса")
            st.pyplot(fig)

    elif slide == "Заключение":
        st.title("Заключение")
        st.write("""
            В этом проекте мы показали:
            - Использование данных для предсказания спроса.
            - Применение методов машинного обучения для улучшения точности прогноза.
            - Потенциальные улучшения для дальнейшего исследования: добавление новых признаков, использование других моделей.
        """)
        st.write("Спасибо за внимание!")
        st.image("path_to_thank_you_image.png")  # Заключительное изображение


if __name__ == "__main__":
    main()