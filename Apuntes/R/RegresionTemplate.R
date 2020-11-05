
# Platilla de Regresión

# Se carga el conjunto de datos

dataset = read.csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
# Filtrado de columnas si es necesario

dataset = dataset[, 2:3]



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

# Ajustar modelo de Regresión

#Crear el modelo de regresión aqui regression = ... 



# Prediccion de nuevos resultados con Regresion
y_pred = (predict(regression, newdata = data.frame(Level = 6.5)))

# Visualizacion del modelo de Regresión
library(ggplot2)
ggplot() + 
        geom_point(aes(x = dataset$Level, 
                       y = dataset$Salary), colour =  "red") +
        geom_line(aes(x=dataset$Level, 
                      y = predict(regression, newdata = dataset)), colour = "blue") +
        ggtitle("Modelo de Regresión") +
        xlab("Nivel del empleado")+
        ylab("Sueldo en USD")