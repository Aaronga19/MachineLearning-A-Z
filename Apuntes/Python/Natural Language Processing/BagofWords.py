# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:33:31 2020

@author: Aaronga
"""

"""                                       Bag of Words (NLP)                                          """

# Importamos las librerías 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dataset
dataset = pd.read_csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv", sep = "\t", quoting = 3)


# Limpeza de texto
import re 
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for rvw in range(0,len(dataset)):
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][rvw]) # [^a-zA-Z] Expresiones que quiero mantener
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))] 
    review = " ".join(review)
    corpus.append(review)
"""Hasta este punto se han trabajado las cadenas de carácteres que necesitamos para el algoritmo,
por lo que se removieron palabras del stopsword y convirtiendo los verbos a infinitivo (conviriendo las
cadenas en conjuntos de datos y al final volver a concatenar)"""

# Crear Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset["Liked"].values

# A partir de aqui se usal algoritmos de Clasificación (Principalmente: Kernel SVM, Naive Bayes, Trees)
""" Por Naive Bayes """
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
                       
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Ajustar el clasificador
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(XTrain,YTrain)

# Predicción del Clasificador 


y_pred = classifier.predict(XTest)

# Elaborar una matriz de confusión 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(YTest, y_pred) # Así podemos tener una breve descripcion de como es el modelo, en donde acertó y cuantas veces se equivocó 

                                                                                        
good = []
bad = []
for i in dataset["Liked"]:
    if i == 1:
        good.append(i)
    else:
        bad.append(i)
print(f"""Se tuvieron {len(good)} reseñas buenas. 
      Y se tuvieron {len(bad)} reseñas malas.""")
                                                                                            



