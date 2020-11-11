# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:54:43 2020

@author: Aaronga
"""

#Importación de librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargar el conjunto de datos y dividirlos por variable independiente variable(s) independientes (s)

dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values # Para convertirlo a vector se usa [:,1:2]
y = dataset.iloc[:, 2].values

# Dicidir el data set en conjunto de entrenamiento y conjunto de testing
 """                       
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables 
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
XTrain = sc_X.fit_transform(XTrain)
XTest = sc_X.transform(XTest)
"""

# Ajustar regresion CORRESPONDIENTE

"""Aplicar aqui el modelo de Regresión"""

# Predicción correspondiente


y_pred = regression.predict([[6.5]])

# Visualizacion de los datos del modelo 

X_grid = np.arange(min(X), max(X+0.1), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
X_polyGrid = polyReg.fit_transform(X_grid)
plt.scatter(X,y, color= "red")
plt.plot(X_grid, regression.predict(X_grid), color="blue") # Parametro original X_poly 
plt.title("Modelo de Regresión")
plt.xlabel("Nivel en la empresa")
plt.ylabel("Salario")
plt.show()
