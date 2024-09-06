# -*- coding: utf-8 -*-

# flake8: noqa
# noqa: E501


import numpy as np
import matplotlib.pyplot as plt

# Генерация синтетических данных
np.random.seed(0)
g_X = 2 * np.random.rand(100, 1)
g_y = 4 + 3 * g_X + np.random.randn(100, 1)


# Градиентный спуск для простой линейной регрессии
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    m = len(y)
    X_b = np.c_[
        np.ones((m, 1)), X
    ]  # Добавление вектора единиц для свободного члена (bias term)
    theta = np.random.randn(
        2, 1
    )  # Инициализация случайных значений для коэффициентов (theta)

    for _iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # Вычисление градиента
        theta = theta - learning_rate * gradients  # Обновление коэффициентов

    return theta


# Выполнение градиентного спуска
theta_final = gradient_descent(g_X, g_y)

# Прогнозирование с использованием полученных коэффициентов
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Добавление вектора единиц для прогнозирования
y_predict = X_new_b.dot(theta_final)

# Визуализация
plt.scatter(g_X, g_y, color="black", label="Данные")
plt.plot(X_new, y_predict, color="red", label="Градиентный спуск")
plt.xlabel("Признак X")
plt.ylabel("Значение Y")
plt.title("Линейная регрессия с использованием градиентного спуска")
plt.legend()
plt.show()

print("Коэффициенты после градиентного спуска:", theta_final.ravel())
