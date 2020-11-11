# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:51:54 2020

@author: Aaronga
"""

# Maquinas de soporte vectorial (SVR

#Importación de librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargar el conjunto de datos y dividirlos por variable independiente variable(s) independientes (s)

dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values # Para convertirlo a vector se usa [:,1:2]
y = dataset.iloc[:, 2].values

# Dicidir el data set en conjunto de entrenamiento y conjunto de testing
 """                       
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Ajustar regresion CORRESPONDIENTE

from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X,y)
# Predicción correspondiente

y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))
y_pred = sc_y.inverse_transform(y_pred)
#print(sc_X.fit_transform(X.reshape(-1,1)))

import statsmodels.api as sm

# Visualizacion de los datos del modelo 

X_grid = np.arange(min(X), max(X+0.1), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
X_polyGrid = polyReg.fit_transform(X_grid)
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y), color= "red") # Valores reales
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regression.predict(X)), color="blue") # Parametro original X_poly 
plt.title("Modelo de Regresión")
plt.xlabel("Nivel en la empresa")
plt.ylabel("Salario")
plt.show()


# Con las variables escaladas

plt.scatter(X,y, color= "red")
plt.plot(X, regression.predict(X), color="blue") #