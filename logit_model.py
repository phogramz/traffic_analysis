# Оценка Факторной нагрузки с помощью Логистической регрессии (или логит-модели)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# Загрузка данных из Excel файла
excel_file = "logit_data_1600.xlsx"
data = pd.read_excel(excel_file, sheet_name="data")
test_data = pd.read_excel(excel_file, sheet_name="test_data")

# Убедимся, что нужные столбцы есть
required_columns = {"Y", "X1", "X2", "X3"}
if not required_columns.issubset(data.columns) or not required_columns.issubset(test_data.columns):
    raise ValueError(f"В одном из листов отсутствуют обязательные столбцы: {required_columns}")

print("\n=======Выводим первые 5 строк=======\n")
print(data.head(5))  # Проверяем первые 5 строк

# Добавляем константу для смещения (intercept)
X_train = sm.add_constant(data[["X1", "X2", "X3", "X22"]]) # Предикторы + константа
Y_train = data["Y"]

X_test = sm.add_constant(test_data[["X1", "X2", "X3", "X22"]]) # Предикторы + константа
Y_test = test_data["Y"]

# X_train = X_train.drop(columns=["X3"]) # Временно убрать временной признак
# X_test = X_test.drop(columns=["X3"]) # Временно убрать временной признак

# Обучаем логистическую регрессию
model = sm.Logit(Y_train, X_train)
result = model.fit()

# Выводим сводку
print(result.summary())


"""Тестирование модели"""
# Предсказания
y_pred_probs = result.predict(X_test)
y_pred = (y_pred_probs > 0.25).astype(int)

print(f"y_pred_probs: {y_pred_probs}")
print(f"y_pred: {y_pred}")

# Вычисление метрик
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred, zero_division=0)
recall = recall_score(Y_test, y_pred, zero_division=0)
f1 = f1_score(Y_test, y_pred, zero_division=0)

# Строим ROC кривую
roc_auc = roc_auc_score(Y_test, y_pred_probs)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_probs)

# Визуализация ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.8f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print("\nМетрики качества модели:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print(f"ROC AUC:   {roc_auc:.8f}")


# Выводим коэффициенты регрессии
result.summary()
# Экспоненцирование коэффициентов модели (Odds Ratio)
print("\n=======Odds Ratio коэффициентов модели=======\n")
print(np.exp(result.params))
# Предсказания модели
print("\n=======Предсказания модели=======\n")
predictions = result.predict(X_train)


plt.hist(y_pred_probs, bins=20)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Distribution of predicted probabilities")
plt.show()


# График предсказания
plt.figure(figsize=(8, 5))
plt.scatter(range(len(predictions)), predictions, label='Predicted Probability', color='blue', alpha=0.6)
plt.axhline(y=0.35, color='red', linestyle='--', label='Decision Boundary (0.35)')
plt.xlabel("Observation Index")
plt.ylabel("Predicted Probability")
plt.title("Predictions of Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
