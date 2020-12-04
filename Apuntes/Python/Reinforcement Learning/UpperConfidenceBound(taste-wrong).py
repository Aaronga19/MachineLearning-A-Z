# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:19:43 2020

@author: Aaronga
"""

"                                      Upper confidence bound                                       """

# Importar librerías 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar dataset
dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

" Ejecutar random_selection.py "
# Algoritmo de Upper Confidence Bound
import math
N = 10000 # Número de rondas 
d = 10 # Número de anuncios 
numberOfSelections = [0] * d
rewardsSum = [0] * d
addsSelected = []
totalReward = 0
for n in range(0, N):
    maxUpperBound = 0
    ad = 0
    for i in range(0, d):
        if(numberOfSelections[i]>0):
            averageReward = rewardsSum[i] / numberOfSelections[i]
            delta = math.sqrt(3/2*math.log(n+1)/numberOfSelections[i])
            upperBound = averageReward + delta
        else:
            upperBound = 1e400
            
        if upperBound > maxUpperBound:
            maxUpperBound = upperBound
            ad = i
    addsSelected.append(ad)
    numberOfSelections[ad] = numberOfSelections[ad] + 1
    reward = dataset.values[n, ad]
    rewardsSum[ad] = rewardsSum[ad] + reward
    totalReward = totalReward + reward
    
# Histograma de resultados
plt.hist(addsSelected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()
    