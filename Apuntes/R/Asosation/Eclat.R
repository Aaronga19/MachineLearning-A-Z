



#                               Eclat

# Preprocesado de datos 
dataset <- read.csv("GitHub/MachineLearningAZ/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv", header = F)
read.transactions("GitHub/MachineLearningAZ/datasets/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = T) -> dataset

summary(dataset)
# Matrices formales

#install.packages("arules")
library(arules)
itemFrequencyPlot(dataset, topN = 100)
itemFrequencyPlot(dataset, topN = 10)

# Entrenar algoritmo de Eclat con el dataset
default = 0.8
rules = eclat(dataset, parameter = list(support= round((3*7)/(7500), digits = 3), minlen = 2))
# Vizualisación de los datos
inspect(sort(rules, by= "support")[1:10])

install.packages("arulesViz")
library(arulesViz)

plot(rules, method = "graph", engine = "htmlwidget")