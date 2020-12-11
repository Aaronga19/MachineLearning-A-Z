# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:59:41 2020

@author: Aaronga
"""

"""                                  Bosques Aleatorios                                            """

# Se importa el Dataset__

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargar el conjunto de datos y dividirlos por variable independiente variable(s) independientes (s)

dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 3 - Classification\Section 14 - Logistic Regression/Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values # Para convertirlo a vector se usa [:,1:2]
y = dataset.iloc[:, 4].values

# Dicidir el data set en conjunto de entrenamiento y conjunto de testing
                       
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Escalado de variables 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
XTrain = sc_X.fit_transform(XTrain)
XTest = sc_X.transform(XTest)


# Ajustar el clasificador Random Forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,
                                    criterion= "entropy",
                                    random_state=0)
classifier.fit(XTrain, YTrain)

# Predicción del Clasificador 


y_pred = classifier.predict(XTest)

# Elaborar una matriz de confusión 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(YTest, y_pred) # Así podemos tener una breve descripcion de como es el modelo, en donde acertó y cuantas veces se equivocó 

accuracy = (cm[0][0]+cm[1][1])/sum(sum(cm))
print(f"The accuracy is: {accuracy:>2.3f} = {accuracy*100:>2.1f}%")
# Visualizacion de los datos del Clasificador

from matplotlib.colors import ListedColormap
X_set, y_set = XTrain, YTrain
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop=X_set[:,0].max()+1, step= 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop=X_set[:,1].max()+1, step= 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(X_set[y_set== j,0], X_set[y_set==j,1],  
                c = ListedColormap(('red','green'))(i),label = j)
plt.title("Random Forest (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()



X_set, y_set = XTest, YTest
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop=X_set[:,0].max()+1, step= 0.01),
                     np.arange(start = X_set[:,1].min()-1,stop=X_set[:,1].max()+1, step= 0.01))
plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate (np.unique(y_set)):
    plt.scatter(X_set[y_set== j,0], X_set[y_set==j,1],
                c = ListedColormap(('red','green'))(i),label = j)
plt.title("Random Forest (Testing set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()