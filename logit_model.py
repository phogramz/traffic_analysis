# Оценка Факторной нагрузки с помощью Логистической регрессии (или логит-модели)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve

# Фиксируем сид для воспроизводимости
np.random.seed(42)

# Генерируем случайные бинарные значения (0 или 1) для целевой переменной и предикторов
# data = pd.DataFrame({
#     "Y": np.random.randint(0, 2, 50),  # Нарушение (целевая переменная)
#     "X1": np.random.randint(0, 2, 50),  # Группа людей (предиктор 1)
#     "X2": np.random.randint(0, 2, 50),  # Время 12-18 (предиктор 2)
#     "X3": np.random.randint(0, 2, 50),  # Машина рядом (предиктор 3)
# })

# Бинарные
# data = pd.DataFrame({
#     "Y":  [0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1],  # Нарушение
#     "X1": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,0],  # Человек один (нет группы людей)
#     "X2": [1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1],  # Машина рядом
#     "X3": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # Время 12-18
# })

# Не Бинарные
data = pd.DataFrame({
    "Y":  [0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1],  # Нарушение
    "X1": [2,5,3,3,2,2,2,2,3,3,3,3,3,3,3,1,5,5,5,5,5,6,4,5,4,3,4,3,2,2,2,2,2,1,2,2,3,4,3,2,3,1,6,6,6,3,4,6,6,5,5,5,6,6,6,4,4,4,4,4,3,1,2,2,2,1,1,2,2,2,1,1,2,2,5,6,6,6,5,6,4,2,3,2,1,2,3,2,2,6,3,2,3,2,3,1,1,3,3,3,3,3,3,3,2,3,2,2,3,2,2,3,1,5,4,4,4,5,4,3,3,2,5,5,6,7,6,6,6,8,7,7,6,6,5,5,5,5,5,6,5,6,5,1,2,2,1,2,1,1,3,4,4,4,4,5,5,5,6,5,5,6,4,5,5,4,5,6,7,6,4,4,2,1,2,2,3,3,3,4,1,4,3,3,3,4,3,3,4,3,6,5,4,2,1,1,1,1,2,2,3,4,3,4,1,1,3,2,4,3,4,5,2,1,1,1,2,4,5,3,5,3,3,1,2,2,2,1,1,1,2,2,2,4,4,3,5,6,3,3,2,3,1,2,5,5,5,6,6,5,6,7,6,7,6,7,6,6,5,6,6,5,7,5,6,7,8,10,10,9,6,6,7,7,9,6,7,7,6,8,6,4,7,1,1,2,4,5,4,5,5,4,3,3,6,6,7,5,5,3,3,3,3,3,3,1,3,2,3,3,3,3,1,5,5,4,4,3,3,1,2,2,2,2,2,2,1,1,1,2,2,1,2,3,4,3,5,3,2,3,3,1,2,2,1,2,1,2,3,4,4,5,5,3,4,4,5,2,3,2,3,3,4,5,2,4,5,6,6,6,4,2,2,3,3,2,2,1,2,2,4,4,4,4,4,3,5,4,4,4,5,6,5,4,5,5,5,2,3,4,4,2,3,3,1,1,1,3,4,3,3,2,3,1,2,3,2,1,1,2,1,1,6,7,7,7,7,2,7,7,9,8,10,8,6,7,7,6,4,4,4,4,3,5,5,4,6,5,7,4,4,5,4,3,1,3,2,3,3,3,3,2,5,5,5,5,5,6,6,6,7,7,6,7,8,7,1,3,3,4,3,2,2,3,3,3,4,1,1,2,2,2,2,2,1,2,1,1,2,4,6,5,5,5,4,4,3,3,3,3,2,3,3,4,1,2,1,2,2,3,2,1,2,2,3,3,4,2,2,2,2,2,2,2,3,3,1,2,2,2,3,3,4,3,2,1,1,1,3,3,3,5,5,3,3,3,4,3,3,2,3,4,3,4,3,3,4,3,3,2,3,2,2,2,3,3,1,1,3,1,1,2,2,1,2,3,4,3,2,2,3,5,6,6,5,4,5,5,7,8,7,4,3,3,2,2,4,4,4,2,1,3,3,3,4,5,4,3,4,5,5,3,1,1,3,4,1,2,1,1,1,3],  # Человек один (нет группы людей)
    "X2": [2,2,2,2,1,0,0,1,1,2,2,1,1,1,2,1,0,0,0,1,1,1,2,2,1,1,1,2,2,1,1,1,1,2,2,1,2,2,2,1,1,2,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,0,1,1,1,1,0,1,1,2,2,2,2,1,2,2,2,2,1,1,0,0,0,0,0,0,0,0,0,2,3,2,2,3,3,1,1,0,0,0,0,1,1,0,1,2,1,2,2,1,2,2,2,1,1,1,1,1,1,0,0,2,0,0,0,0,0,0,0,0,0,0,2,2,1,1,1,1,1,1,1,1,1,2,2,1,1,2,2,3,3,4,3,3,2,2,2,2,2,1,1,1,1,0,0,1,1,1,1,1,1,2,1,4,2,2,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,3,2,3,2,2,3,2,3,3,2,1,3,0,0,0,0,0,0,1,0,1,0,1,1,1,1,1,1,2,1,1,1,0,0,0,0,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,0,2,2,0,0,2,1,1,2,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,2,2,2,2,2,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,2,2,1,1,1,1,0,4,4,4,2,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,2,2,2,1,1,0,1,1,1,1,0,2,2,2,1,1,3,2,3,2,2,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,0,1,1,1,1,2,1,1,1,1,1,1,1,2,2,2,1,2,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,1,2,2,2,2,2,1,1,1,1,2,2,1,1,1,2,1,1,1,2,1,2,0,1,0,1,1,0,1,1,1,1,1,2,2,2,2,1,1,1,1,2,1,0,1,2,1,1,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,2,3,3,2,2,0,0,0,0,0,1,3,1,2,2,1,1,1,2,2,1,1,1,1,1,1,1,1,1,1,2,2,1,1,1,1,2,1,1,1,2,2,1,1,1,1,1,1,1,2,2,1,1,0,0,0,0,1,1,0,0,1,0,2,2,2,2,2,2,1,1,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1],  # Машина рядом
    "X3": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  # Время 12-18
})

print("\n=======Выводим первые 5 строк=======\n")
print(data.head(5))  # Проверяем первые 5 строк

# scaler = StandardScaler()

# Разделение на признаки и целевую переменную
X_train = data[["X1", "X2", "X3"]]
Y_train = data["Y"]

# Добавляем константу для регрессии
X_train = sm.add_constant(X_train)  # Предикторы + константа
X_train = X_train.drop(columns=['X3']) # Временно убрать временной признак

# X_train_scaled = scaler.fit_transform(X_train)


# Обучаем логистическую регрессию
model = sm.Logit(Y_train, X_train)
result = model.fit()

"""Тестирование модели"""
# Создаем тестовую матрицу
test_data = pd.DataFrame({
    "X1": [2,1,1,1,1,2,4,4,5,1,2,5,5,5,5,5,4,3,4,2,3,4,5,5,6,3,4,5,6,6,7,7,6,7,1,2,3,4,3,3,4,4,3,2,3,3,1,1,1,5,8,8,10,7,8,8,6,7,7,5,4,3,3,2,4,4,5,3,3,3,3,3,3,4,4,3,3,2,3,3,2,6,6,5,6,5,5,5,4,4,5,6,3,4,2,3,3,4,4,5,4,6,6,6,6,6,7,5,2,3,3,2,2,2,2,1,1,2,2,3,4,4,4,3,4,4,3,2,3,4,1,2,1,1,2,4,5,4,4,5,2,3,3,3,3,3,1,2,2,4,2,3,1,4,5,4,3,3,3,2,4,1,1,1,2,3,3,1,2,2,3,1,2,2,1,2,2,1,2,2,3,2,1,2,2,2,1,2,1,1,2,3,3,3,1,1,2,2,2,2,2,2,2,4,3,5,5,4,4,4,4,5,6,1,2,2,2,4,3,2,2,2,3,3,3,4,4,5,4,4,2,2,2,1,2,1,3,5,4,4,4,4,5,5,5,6,5,2,4,5,4,4,3,3,5,3,4,4,7,7,8,6,6,6,7,7,7,8,7,7,6,6,5,6,5,4,4,4,5,5,6,6,6,6,6,7,6,7,4,6,2,3,2,2,2,2,2,2,3,3],
    "X2": [0,0,0,1,2,2,1,1,1,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2,2,0,0,0,0,0,0,0,1,1,1,1,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,2,1,3,2,1,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,0,0,0,0,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,1,1,1,1,0,0,0,0,0,1,1,2,2,2,2,2,2,2,2,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,2,1,1,2,2,2,2,2,1,1,1,1,1,0,1,1,0,0,1,1,0,0,2,2,2,2,2,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,0,2,2,0,2,1,1,1,1,2,2,1,1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,0,0,0,0],
    "X3": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
})

# Добавляем константу
X_test = sm.add_constant(test_data)
X_test = X_test.drop(columns=['X3']) # Временно убрать временной признак

# X_test_scaled = scaler.transform(X_test)

# Предсказания
pred_probs = result.predict(X_test)
pred_labels = (pred_probs > 0.35).astype(int)  # Бинаризация

# Для расчетов метрик добавляем истинные значения (их нужно заполнить вручную)
Y_test = [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]  # Пока все 0, но можно поменять вручную

# Вычисление метрик
accuracy = accuracy_score(Y_test, pred_labels)
precision = precision_score(Y_test, pred_labels, zero_division=0)
recall = recall_score(Y_test, pred_labels, zero_division=0)
f1 = f1_score(Y_test, pred_labels, zero_division=0)


print(f"pred_probs: {pred_probs}")
print(f"pred_labels: {pred_labels}")

print(test_data.describe())
# Вывод результатов
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Выводим коэффициенты регрессии
result.summary()

# Коэффициенты модели
print("\n=======Коэффициенты модели=======\n")
print(result.params)

# Экспоненцирование коэффициентов модели (Odds Ratio)
print("\n=======Odds Ratio коэффициентов модели=======\n")
print(np.exp(result.params))

print("\n=======Предсказания модели=======\n")
# Предсказания модели
predictions = result.predict(X_train)
print("\n=================================\n")


################################################################



# Предсказанные вероятности для положительного класса
p_pred_probs = result.predict(X_test)

# Реальные метки классов (0 или 1)
Y_true = Y_test

# Рассчитываем AUC
auc = roc_auc_score(Y_true, p_pred_probs)
print(f'ROC AUC: {auc}')

# Строим ROC кривую
fpr, tpr, thresholds = roc_curve(Y_true, p_pred_probs)

# Визуализация ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


















plt.hist(pred_probs, bins=20)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of predicted probabilities")
plt.show()



# График
plt.figure(figsize=(8, 5))
plt.scatter(range(len(predictions)), predictions, label='Predicted Probability', color='blue', alpha=0.6)
plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Boundary (0.5)')
plt.xlabel("Observation Index")
plt.ylabel("Predicted Probability")
plt.title("Predictions of Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
