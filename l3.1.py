# -*- coding: utf-8 -*-

# flake8: noqa
# noqa: E501

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Генерация синтетических данных
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Создание модели линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Прогнозирование
y_pred = model.predict(X)

# Оценка модели
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean squared error:", mse)
print("R2 score:", r2)

# Визуализация данных и предсказаний
plt.scatter(X, y, color="black", label="Данные")
plt.plot(X, y_pred, color="blue", linewidth=3, label="Линейная регрессия")
plt.xlabel("Признак X")
plt.ylabel("Значение Y")
plt.title("Линейная регрессия: пример сгенерированных данных")
plt.legend()
plt.show()
