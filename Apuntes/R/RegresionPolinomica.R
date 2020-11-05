
# Regresión polinomica

dataset = read.csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
dataset = dataset[, 2:3]

## Tratamiento de los NaN
"""
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x)mean(x,na.rm = T)),dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x)mean(x,na.rm = T)),dataset$Salary)
"""

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

trainingSet[,2:3] = scale(trainingSet[,2:3])
testingSet[,2:3] = scale(testingSet[,2:3])

# Ajustar modelo lineal 

linReg = lm(Salary ~.,data = dataset)
summary(linReg)
# Ajustar modelo polinomico
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
dataset$Level5 = dataset$Level^5

polyReg = lm(Salary ~ ., data = dataset)
summary(polyReg)


# Prediccion de nuevos resultados con regresion lineal
y_pred = (predict(linReg, newdata = data.frame(Level = 6.5)))

# Prediccion de nuevos resultados con regresion polinomica
y_predPoly = (predict(polyReg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4,
                                                Level5 = 6.5^5)))

# Visualizacion del modelo lineal 
library(ggplot2)
ggplot() + 
        geom_point(aes(x = dataset$Level, 
                       y = dataset$Salary), colour =  "red") +
        geom_line(aes(x=dataset$Level, 
                      y = predict(linReg, newdata = dataset)), colour = "blue") +
        ggtitle("Predicción lineal del sueldo en funcion del empleado") +
        xlab("Nivel del empleado")+
        ylab("Sueldo en USD")

# Visualización del modelo polinomico
x_grid =  seq(min(dataset$Level), max(dataset$Level),0.1)
ggplot() + 
        geom_point(aes(x = dataset$Level, 
                       y = dataset$Salary), colour =  "red") +
        geom_line(aes(x=x_grid, 
                      y = predict(polyReg, newdata = data.frame(Level = x_grid,
                                                                Level2 = x_grid^2,
                                                                Level3 = x_grid^3,
                                                                Level4 = x_grid^4,
                                                                Level5 = x_grid^5))), colour = "blue") +
        ggtitle("Predicción polinomica del sueldo en funcion del empleado") +
        xlab("Nivel del empleado")+
        ylab("Sueldo en USD")




