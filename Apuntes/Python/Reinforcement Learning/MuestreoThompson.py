# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:28:05 2020

@author: Aaronga
"""

"""                                        Muestreo Thompson                                      """
# Importar librerías 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Cargar dataset
dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

" Ejecutar random_selection.py "
# Algoritmo de Muestreo Thompson 
import random # Librería para generar probabilidades
N = 10000 # Número de rondas 
d = 10 # Número de anuncios 
numberOfRewards1 =  [0]*d
numberOfRewards0 =  [0]*d
rewardsSum = [0] * d
addsSelected = []
totalReward = 0
for n in range(0, N):
    maxRandom = 0
    ad = 0
    for i in range(0, d):
        randomBeta = random.betavariate(numberOfRewards1[i]+1, numberOfRewards0[i]+1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            ad = i
    addsSelected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numberOfRewards1[ad] +=1
    else:
        numberOfRewards0[ad] +=1
    totalReward = totalReward + reward
    
# Histograma de resultados
plt.hist(addsSelected)
plt.title("Histograma de anuncios")
plt.xlabel("ID del Anuncio")
plt.ylabel("Frecuencia de visualización del anuncio")
plt.show()