# -*- coding: utf-8 -*-

# flake8: noqa
# noqa: E501

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Загрузка датасета Iris
iris = load_iris()
X, y = iris.data, iris.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создание и обучение модели дерева решений
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred_dt = dt.predict(X_test)

# Оценка модели
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Визуализация дерева решений
plt.figure(figsize=(12, 8))
tree.plot_tree(
    dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True
)
plt.show()
