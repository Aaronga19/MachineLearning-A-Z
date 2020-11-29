# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:32:54 2020

@author: Aaronga
"""

"""                                         Cluster Jerárquico                                      """

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos con pandas
dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

# Utilizar el dendograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendograma")
plt.ylabel("Distancia Euclidea")
plt.xlabel("Clientes")
plt.show()                  # Salen 5 clusters aquí 


# Ajustar el Clustering Jerárquico a nuestro conjunto de datos 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity= "euclidean", linkage="ward")
y_hc = hc.fit_predict(X)


# Visualización de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Estandard")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "brown", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()