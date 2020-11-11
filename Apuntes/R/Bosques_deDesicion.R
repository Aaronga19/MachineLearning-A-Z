# Regresión de Bosques Aleatorios

# Se carga el conjunto de datos

dataset = read.csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
dataset = dataset[, 2:3] # Filtrado de columnas si es necesario

# codificar variables categoricas

"""
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No","Yes"),
                           labels = c(0,1))
"""

# Dividir los datos en conjunto de entrenamiento y conjunto de test
"""
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
split
trainingSet = subset(dataset, split == T)
testingSet = subset(dataset, split == F)
"""

# Escalado de valores 
"""
trainingSet[,2:3] = scale(trainingSet[,2:3])
testingSet[,2:3] = scale(testingSet[,2:3])
"""

# Ajustar modelo de Regresión Random Forest
install.packages("randomForest")
library(randomForest)
set.seed(1234)
regression = randomForest(x = dataset[1], 
                          y = dataset$Salary, 
                          ntree = 300) 



# Prediccion de nuevos resultados con Regresion Random Forest
y_pred = (predict(regression, newdata = data.frame(Level = 6.5)))

# Visualizacion del modelo de Regresión Random Forest
library(ggplot2)
x_grid =  seq(min(dataset$Level), max(dataset$Level),0.01)
ggplot() + 
        geom_point(aes(x = dataset$Level, 
                       y = dataset$Salary), colour =  "red") +
        geom_line(aes(x=x_grid, 
                      y = predict(regression, newdata = data.frame(Level = x_grid))), colour = "blue") +
        ggtitle("Modelo de Regresión de Bosque de desición") +
        xlab("Nivel del empleado")+
        ylab("Sueldo en USD")