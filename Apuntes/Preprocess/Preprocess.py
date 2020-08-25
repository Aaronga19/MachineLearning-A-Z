# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:05:59 2020

@author: Aaronga
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

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
y=labelEncoderY.fit_transform(y)

                    # VARIABLE DUMMY 
                        # 0 0 0/ 0 1 0 / 0 0 1
# Dicidir el data set en conjunto de entrenamiento y conjunto de testing
                        
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
XTrain = sc_X.fit_transform(XTrain)
XTest = sc_X.transform(XTest)