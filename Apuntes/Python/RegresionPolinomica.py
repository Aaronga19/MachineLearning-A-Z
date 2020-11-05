# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:30:41 2020

@author: Aaronga
"""

# Regresión polinómica

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values # Para convertirlo a vector se usa [:,1:2]
y = dataset.iloc[:, 2].values
"""
# Tratamiento de los NaN 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3]= imputer.transform(X[:,1:3])
print(X)

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:,0]=labelEncoderX.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelEncoderY = LabelEncoder()
y=labelEncoderY.fit_transform(y)"""

                    # VARIABLE DUMMY 
                        # 0 0 0/ 0 1 0 / 0 0 1
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

# Ajustar regresion lineal

from sklearn.linear_model import LinearRegression
linRegression = LinearRegression()
linRegression.fit(X, y)

# Ajustar regresion polynomial

from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=5)
X_poly = polyReg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualización de los datos del modelo lineal
plt.scatter(X,y, color= "red")
plt.plot(X, linRegression.predict(X), color="blue")
plt.title("Salario de acuerdo al nivel (Lineal)")
plt.xlabel("Nivel en la empresa")
plt.ylabel("Salario")
plt.show()

# Visualizacion de los datos del modelo polinomico

X_grid = np.arange(min(X), max(X+0.1), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
X_polyGrid = polyReg.fit_transform(X_grid)
plt.scatter(X,y, color= "red")
plt.scatter(6.5,lin_reg2.predict(polyReg.fit_transform([[6.5]])), color= "green" )
plt.plot(X_grid, lin_reg2.predict(X_polyGrid), color="blue") # Parametro original X_poly 
plt.title("Salario de acuerdo al nivel (Polinomica)")
plt.xlabel("Nivel en la empresa")
plt.ylabel("Salario")
plt.show()

# Predicción correspondiente

linRegression.predict([[6.5]])

lin_reg2.predict(polyReg.fit_transform([[6.5]]))