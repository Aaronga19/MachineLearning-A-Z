# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:27:16 2020

@author: Aaronga
"""

# Regresion lineal Simple

# Se importan las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Se importa el DATASET
SalaryData = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
X = SalaryData.iloc[:, :-1].values
y = SalaryData.iloc[:, 1].values


# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state = 0)

# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Crear modelo de regresión
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento 
plt.scatter(X_train, y_train, color = "red")
plt.scatter(X_test, y_test, color = "pink")
plt.plot(X_train, regression.predict(X_train), color= "blue")
plt.title("Sueldo vs Años de Experiencia(Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario USD") 
plt.show()

plt.scatter(X_train, y_train, color = "red", label = True)
plt.scatter(X_test, y_test, color = "pink")
plt.plot(X_train, regression.predict(X_train), color= "blue")
plt.title("Sueldo vs Años de Experiencia(Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario USD") 
plt.show()