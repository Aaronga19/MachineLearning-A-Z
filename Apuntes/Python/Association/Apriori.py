# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:00:28 2020

@author: Aaronga
"""

"""                                      Apriori                                                      """

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar dataset

dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# Ajustar el modelo 
from apyori import apriori 
rules = apriori(transactions, min_support = round((3*7)/(len(dataset)-1), ndigits= 3), 
                min_confidence = 0.2, min_lift = 3, min_length = 2 )

# Visualización de los datos 
results = list(rules)

results[0]