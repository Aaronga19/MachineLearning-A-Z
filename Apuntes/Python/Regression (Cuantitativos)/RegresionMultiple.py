# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:50:01 2020

@author: Aaronga
"""
# Regresión lineal multiple 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Se importa el DATASET
startUps = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_startups.csv")
X = startUps.iloc[:, :-1].values
y = startUps.iloc[:, 4].values

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X =onehotencoder.fit_transform(X).toarray()

# Evitar la trampa de las variables ficticias y se elimina la primera columna 
X = X[:, 1:]

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 0)

# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar el modelo de Regresion Lineal multiple con el conjunto de entrenamiento
# Crear modelo de regresión
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de testing

y_pred = regression.predict(X_test)

# ELIMINACION HACIA ATRAS
import statsmodels.api as sm
# Para poder calcular el P-Valor

X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1)
SL= 0.05

X_opt = X[:, [0,1,2,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog= X_opt).fit()

# Consultar el P-valor
regression_OLS.summary()

X_opt = X[:, [0,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3]].tolist()
regression_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regression_OLS.summary()

"""
import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""
# Considerando P-Valor y R2-ajustado 
import statsmodels.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
