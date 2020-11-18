# K- Nearest Neighbors 

# Se carga el conjunto de datos

dataset = read.csv("GitHub/MachineLearningAZ/datasets/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv")
# Filtrado de columnas si es necesario

dataset = dataset[, 3:5]

 

# codificar variables categoricas

"""
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No','Yes'),
                           labels = c(0,1))
"""

# Dividir los datos en conjunto de entrenamiento y conjunto de test

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
split
trainingSet = subset(dataset, split == T)
testingSet = subset(dataset, split == F)


# Escalado de valores 

trainingSet[,1:2] = scale(trainingSet[,1:2])
testingSet[,1:2] = scale(testingSet[,1:2])


# Ajustar modelo de K-NN con el conjunto de entrenamiento 
# Prediccion de nuevos resultados con el conjunto de Testing

library(class)
y_pred = knn(train = trainingSet[,-3],
             test = testingSet[,-3],
             cl = trainingSet[,3],
             k = 5)


# Crear matriz de Confusión 

cm = table(testingSet[,3], y_pred)

# Visualizacion del conjunto de entrenamiento
install.packages("ElemStatLearn")
library(ElemStatLearn)
set = trainingSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = trainingSet[,-3],
             test = grid_set,
             cl = trainingSet[,3],
             k = 5)
plot(set[, -3],
     main = 'K-NN (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualización del conjunto de testing
set = testingSet 
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = trainingSet[,-3],
                      test = grid_set,
                      cl = trainingSet[,3],
                      k = 5)
plot(set[, -3],
     main = 'K-NN (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
