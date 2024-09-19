# -*- coding: utf-8 -*-

# flake8: noqa
# noqa: E501

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загрузка датасета Iris
iris = datasets.load_iris()
X = iris.data  # pylint: disable=no-member  # Признаки
y = iris.target  # pylint: disable=no-member  # Метки классов

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создание и обучение модели логистической регрессии
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность модели: {accuracy:.2f}")
