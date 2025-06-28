# modelo.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import generador_features
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, recall_score, f1_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE

# ------------------------
# 1. Cargar y preprocesar datos
# ------------------------
def cargar_y_preprocesar_datos(path_train):
    data = pd.read_csv(path_train)
    data.drop(columns=['Cliente', 'Negocio', 'Fecha_Compra'], inplace=True)

    # Codificación
    data['Grupo_Edad'] = OrdinalEncoder().fit_transform(data[['Grupo_Edad']])
    data['Tipo_Via'] = LabelEncoder().fit_transform(data['Tipo_Via'])
    data['Horario'] = OrdinalEncoder().fit_transform(data[['Horario']])
    data['Tipo_Tarjeta'] = LabelEncoder().fit_transform(data['Tipo_Tarjeta'])
    data['Categoria_Compra'] = LabelEncoder().fit_transform(data['Categoria_Compra'])
    data['Trabajo'] = LabelEncoder().fit_transform(data['Trabajo'])
    data['Genero'] = data['Genero'].replace({'F': 0, 'M': 1})

    return data

# ------------------------
# 2. Entrenar modelo
# ------------------------
def entrenar_modelo(data):
    X = data.drop('Fraude', axis=1)
    y = data['Fraude']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)
    

    y_pred = modelo.predict(X_val)
    y_proba = modelo.predict_proba(X_val)[:, 1]

    print("\nResultados de Evaluación:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_val, y_pred):.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_val, y_pred))

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('Curva ROC')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Importancia de variables
    importancias = modelo.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importancias})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Importancia de las Características')
    plt.xlabel('Importancia')
    plt.ylabel('Características')
    plt.show()

    return modelo, X.columns

# ------------------------
# 3. Predecir sobre dataset de test
# ------------------------
def predecir_dataset_test(modelo, path_test, columnas_modelo):
    df_test = pd.read_csv(path_test)
    df_test.drop(columns=['Cliente', 'Negocio', 'Fecha_Compra'], inplace=True)

    df_test['Grupo_Edad'] = OrdinalEncoder().fit_transform(df_test[['Grupo_Edad']])
    df_test['Tipo_Via'] = LabelEncoder().fit_transform(df_test['Tipo_Via'])
    df_test['Horario'] = OrdinalEncoder().fit_transform(df_test[['Horario']])
    df_test['Tipo_Tarjeta'] = LabelEncoder().fit_transform(df_test['Tipo_Tarjeta'])
    df_test['Categoria_Compra'] = LabelEncoder().fit_transform(df_test['Categoria_Compra'])
    df_test['Trabajo'] = LabelEncoder().fit_transform(df_test['Trabajo'])
    df_test['Genero'] = df_test['Genero'].replace({'F': 0, 'M': 1})

    df_test = df_test[columnas_modelo]  # Aseguramos el orden y variables iguales

    predicciones = modelo.predict(df_test)
    return predicciones

# ------------------------
# 4. Flujo principal
# ------------------------
if __name__ == '__main__':
    ruta_train = 'C:/Users/leo_2/Documents/tarjetas_creditos/data/procesada/data_procesada.csv'
    ruta_test = 'C:/Users/leo_2/Documents/tarjetas_creditos/data/procesada/data_test.csv'  # Ajusta el nombre si es otro

    print("\nCargando y preprocesando datos...")
    data_train = cargar_y_preprocesar_datos(ruta_train)

    print("\nEntrenando modelo...")
    modelo, columnas_modelo = entrenar_modelo(data_train)

    print("\nGenerando predicciones sobre el dataset test...")
    y_pred_test = predecir_dataset_test(modelo, ruta_test, columnas_modelo)

    print("\nPredicciones generadas:")
    print(y_pred_test)
