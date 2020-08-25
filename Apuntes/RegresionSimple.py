# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:27:16 2020

@author: Aaronga
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SalaryData = pd.read_csv("../datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values