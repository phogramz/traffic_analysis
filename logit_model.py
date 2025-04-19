# Оценка Факторной нагрузки с помощью Логистической регрессии (или логит-модели)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


# Загрузка данных из Excel файла
excel_file = "logit_data_mainrand_x1600.xlsx" #logit_data_1600_51_rand logit_data_1600_51_semirand2 logit_data_mainrand logit_data_mainrand_x1600.xlsx
data = pd.read_excel(excel_file, sheet_name="data")
test_data = pd.read_excel(excel_file, sheet_name="test_data")

# Убедимся, что нужные столбцы есть
required_columns = {"Y", "X1", "X2", "X3"}
if not required_columns.issubset(data.columns) or not required_columns.issubset(test_data.columns):
    raise ValueError(f"В одном из листов отсутствуют обязательные столбцы: {required_columns}")

print("\n=======Выводим первые 5 строк=======\n")
# print(data.head(5))  # Проверяем первые 5 строк

# Добавляем константу для смещения (intercept)
X_train = sm.add_constant(data[["X1", "X2", "X3"]]) # Предикторы + константа
Y_train = data["Y"]

X_test = sm.add_constant(test_data[["X1", "X2", "X3"]]) # Предикторы + константа
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
y_pred = (y_pred_probs > 0.145).astype(int)

print(f"y_pred_probs: {y_pred_probs}")
print(f"y_pred: {y_pred}")

# Вычисление метрик
accuracy = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred, zero_division=0)
recall = recall_score(Y_test, y_pred, zero_division=0)
f1 = f1_score(Y_test, y_pred, zero_division=0)

# Вычисление confusion matrix
cm = confusion_matrix(Y_test, y_pred)
# Преобразуем в таблицу
cm_df = pd.DataFrame(cm,
                     index=['Actual: 0', 'Actual: 1'],
                     columns=['Predicted: 0', 'Predicted: 1'])
print("Confusion Matrix:")
print(cm_df)

print("\nМетрики качества модели:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Расчёт Precision-Recall кривой
precisions, recalls, thresholds_pr = precision_recall_curve(Y_test, y_pred_probs)
avg_prec = average_precision_score(Y_test, y_pred_probs)
print(f"Precision-Recall:   {avg_prec:.8f}")

# Расчёт ROC кривой
roc_auc = roc_auc_score(Y_test, y_pred_probs)
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_probs)
print(f"ROC AUC:   {roc_auc:.8f}")

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

# Baseline — доля положительных примеров
baseline_positive_rate = Y_test.mean()

# Визуализация Precision-Recall кривой
plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, color='purple', lw=2, label=f'PR curve (AP = {avg_prec:.4f})')
plt.hlines(y=baseline_positive_rate, xmin=0, xmax=1, color='gray', linestyle='--', label='Baseline')
plt.scatter(recall, precision, color='red', label=f'Threshold = 0.25\n(Prec={precision:.2f}, Rec={recall:.2f})', zorder=5)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

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
plt.axhline(y=0.14, color='red', linestyle='--', label='Decision Boundary (0.14)')
plt.xlabel("Observation Index")
plt.ylabel("Predicted Probability")
plt.title("Predictions of Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()
