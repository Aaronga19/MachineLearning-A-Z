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

""" Por SVM Kernel   """
            
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
                       
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.15, random_state = 0)


# Ajustar modelo Kernel SVM

from sklearn.svm import SVC
classifier = SVC(kernel="poly", random_state=0, gamma=0.5, coef0=2.3, degree=2)
classifier.fit(XTrain,YTrain)

# Predicción SVM

y_pred = classifier.predict(XTest)

# Elaborar una matriz de confusión 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(YTest, y_pred)   

# Cálculo eficacia de algoritmo
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

""" Por Random Forest """ # WIIIIIIIIIIINEEER

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
                       
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Ajustar el clasificador Random Forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500,
                                    criterion= "entropy",
                                    random_state=0,
                                    max_depth= 16,
                                    max_features = "log2")
classifier.fit(XTrain, YTrain)

# Predicción del Clasificador 


y_pred = classifier.predict(XTest)

# Elaborar una matriz de confusión 
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(YTest, y_pred) # Así podemos tener una breve descripcion de como es el modelo, en donde acertó y cuantas veces se equivocó 

# Cálculo eficacia de algoritmo
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

""" Por K-NN """

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Ajustar el clasificador

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10, p = 2, metric="minkowski",
                                  algorithm="brute", leaf_size = 100, n_jobs = 30)
classifier.fit(XTrain,YTrain)

# Predicción del Clasificador 

y_pred = classifier.predict(XTest)

# Elaborar una matriz de confusión 

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(YTest, y_pred)

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


""" CART """
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Ajustar clasificador
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0, splitter = "best",
                                    max_depth= None)
classifier.fit(XTrain, YTrain)

# Predicción del Clasificador 

y_pred = classifier.predict(XTest)

# Elaborar una matriz de confusión 

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(YTest, y_pred)

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

""" MAXENT """
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.20, random_state = 0)
dataTrain = np.column_stack((XTrain, YTrain))

# Ajustar clasificador
def trainMaxent(featuresets):
    #idx = 2*len(featuresets) / ratio
    #train_set, test_set = featuresets[idx:], featuresets[:idx]
    train_set = featuresets
    algo = nltk.classify.MaxentClassifier.ALGORITHMS[1]
    #max_iter=20
    classifier = nltk.classify.MaxentClassifier.train(train_set, algo, max_iter=3)
    #print accuracy(classifier, test_set)
    classifier.show_most_informative_features(100)
    #train_set, test_set = featuresets[idx:], featuresets[:idx]
    #classifier.train(train_set, algo, max_iter=20)
    #print accuracy(classifier, test_set)
    #classifier.show_most_informative_features(100)
    return classifier
trainMaxent(YTrain)
"""---------------------------"""
training_tweets = nltk.classify.get_training_tweets(dataTrain)
training_feature_set = nltk.classify.process_tweets(training_tweets)
classifier = nltk.classify.MaxentClassifier.train(training_feature_set,algorithm="megam", trace=0, max_iter=1000)
# Predicción del Clasificador 

y_pred = classifier.predict(XTest)

# Elaborar una matriz de confusión 

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(YTest, y_pred)

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

"""=============================================================================================="""                                                                         
good = []
bad = []
for i in dataset["Liked"]:
    if i == 1:
        good.append(i)
    elif i == 0:
        bad.append(i)
print(f"""Se tuvieron {len(good)} reseñas buenas. 
      Y se tuvieron {len(bad)} reseñas malas.""")
                                                                                            
from nltk.classify import maxent
train = [
     ({'a': 1, 'b': 1, 'c': 1}, 'y'),
     ({'a': 5, 'b': 5, 'c': 5}, 'x'),
     ({'a': 0.9, 'b': 0.9, 'c': 0.9}, 'y'),
     ({'a': 5.5, 'b': 5.4, 'c': 5.3}, 'x'),
     ({'a': 0.8, 'b': 1.2, 'c': 1}, 'y'),
     ({'a': 5.1, 'b': 4.9, 'c': 5.2}, 'x')
 ]
test = [
     {'a': 1, 'b': 0.8, 'c': 1.2},
     {'a': 5.2, 'b': 5.1, 'c': 5}
]
encoding = maxent.TypedMaxentFeatureEncoding.train(train, count_cutoff=3, alwayson_features=True)
classifier = maxent.MaxentClassifier.train(train, bernoulli=False, encoding=encoding, trace=0)
classifier.classify_many(test)
['y', 'x']