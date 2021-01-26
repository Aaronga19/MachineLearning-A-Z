
#                                                APC

# Se carga el conjunto de datos
dataset = read.csv("GitHub/MachineLearningAZ/datasets/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv")


# codificar variables categoricas

'''
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No","Yes"),
                           labels = c(0,1))
'''

# Dividir los datos en conjunto de entrenamiento y conjunto de test

library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.80)
trainingSet = subset(dataset, split == T)
testingSet = subset(dataset, split == F)


# Escalado de valores 

trainingSet[,-14] = scale(trainingSet[,-14])
testingSet[,-14] = scale(testingSet[,-14])

# Reducir dimension por PCA
install.packages("caret")
library(caret)
library(e1071)
pca = preProcess(x = trainingSet[,-14], method = "pca", pcaComp = 2)
trainingSet= predict(pca,trainingSet)
trainingSet = trainingSet[,c(2,3,1)]
testingSet= predict(pca,testingSet)
testingSet = testingSet[,c(2,3,1)]


# Ajustar modelo de Regresión Logística con el conjunto de entrenamiento 
    # Por Regresión Logística
classifier = glm(formula = Customer_Segment ~ ., 
                 data = trainingSet,
                 family = binomial)

    # Por Support Vector Machine
classifier = svm(formula = Customer_Segment ~ ., 
                 data = trainingSet,
                 type = "C-classification",
                 kernel = "linear")


# Prediccion de nuevos resultados con el conjunto de Testing
    # Regresión logística
prob_pred = predict(classifier, type = "response", newdata = testingSet[,-3])
y_pred = ifelse(prob_pred>0.5,1,0)

    # Support Vector Machine
y_pred = predict(classifier, newdata = testingSet[,-3])


# Crear matriz de Confusión 

cm = table(testingSet[,3], y_pred)

# Visualizacion del conjunto de entrenamiento
library(ElemStatLearn)
set = trainingSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2,'deepskyblue',
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[,3]==2, 'blue3',
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualización del conjunto de testing
set = testingSet
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 2,'deepskyblue',
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[,3]==2, 'blue3',
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))