# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:18:37 2021

@author: Aaronga
"""

#                                         Deep Learning
"""                               Redes Neuronales Artificiales                                       """


# Conda install -c conda-forge keras
# Instalar Theano


# Se importa el Dataset__

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargar el conjunto de datos y dividirlos por variable independiente variable(s) independientes (s)

dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values # Para convertirlo a vector se usa [:,1:2]
y = dataset.iloc[:, 13].values

#                    Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderXPais = LabelEncoder()
X[:,1]=labelEncoderXPais.fit_transform(X[:,1])
labelEncoderXGenero = LabelEncoder()
X[:,2]=labelEncoderXGenero.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
                       
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.20, random_state = 0)

# OBLIGATORIO!
# Escalado de variables 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
XTrain = sc_X.fit_transform(XTrain)
XTest = sc_X.transform(XTest)

"""                                    Construir RED NEURONAL                                      """
# Importar Keras 
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la red nueronal artificial (RNA/Arquitectura)
classifier = Sequential() # Capas (Dendritas)

# Añadir las capas de entrada y la primer capa oculta
classifier.add(Dense(units=6,kernel_initializer='uniform', activation="relu", input_dim = 11)) 
# Añadir segunda capa oculta
classifier.add(Dense(units=6,kernel_initializer='uniform', activation="relu")) # La capa ya sabe que ahora la entrada será de 6

# Añadir capa de salida
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


# Compilar RNA
classifier.compile(optimizer='adam',loss="binary_crossentropy", metrics=["accuracy"])


# Ajustar la RNA al conjunto de entrenamiento
classifier.fit(XTrain, YTrain, batch_size=10, epochs=100)

"""Crear el modelo de clasificación aqui"""

# Predicción del Clasificador 

y_pred = classifier.predict(XTest) #Se obtienen probabilidades
getout = []
for prob in y_pred:
    if prob > 0.8:
        getout.append(1)
    elif prob >=0.6:
        getout.append(0.5)
    else:
        getout.append(0)
        
y_pred= (y_pred>0.5)
# Elaborar una matriz de confusión 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(YTest, y_pred) # Así podemos tener una breve descripcion de como es el modelo, en donde acertó y cuantas veces se equivocó 

# Cálculo eficacia de algorítmo
accuracy = (cm[0][0]+cm[1][1])/sum(sum(cm))
precision = cm[0][0]/(cm[0][0]+cm[0][1])
recall = cm[0][0]/(cm[0][0]+cm[1][0])
f1Score = (2*precision*recall)/(precision+recall)
print(f"""
      The accuracy is: {accuracy:<5.3f} = {accuracy*100:<5.1f}%
      The precision is: {precision:<5.3f} = {precision*100:<5.1f}%
      The recall is: {recall:<5.3f} = {recall*100:<5.1f}%
      The F1Score is: {f1Score:<5.3f} = {f1Score*100:<5.1f}%
      """)