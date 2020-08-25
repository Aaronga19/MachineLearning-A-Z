# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:30:35 2020

@author: Aaronga
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:,0]=labelEncoderX.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelEncoderY = LabelEncoder()
y=labelEncoderY.fit_transform(y)