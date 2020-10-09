
# Regresion Simple

## Tratamiento de los NaN

dataset = read.csv("GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
#dataset = dataset[,2:3]

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x)mean(x,na.rm = T)),dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x)mean(x,na.rm = T)),dataset$Salary)


# codificar variables categoricas


#dataset$Country = factor(dataset$Country,
 #                        levels = c("France", "Spain", "Germany"),
 #                        labels = c(1,2,3))
#dataset$Purchased = factor(dataset$Purchased,
 #                          levels = c("No","Yes"),
 #                          labels = c(0,1))


# Dividir los datos en conjunto de entrenamiento y conjunto de test

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
trainingSet = subset(dataset, split == T)
testingSet = subset(dataset, split == F)


# Escalado de valores 

#trainingSet[,2:3] = scale(trainingSet[,2:3])
#testingSet[,2:3] = scale(testingSet[,2:3])

# Ajustar el modelo de regresion lineal simple con el conjunto de entrenamiento 

regressor = lm(formula = Salary ~ YearsExperience, data = trainingSet)

summary(regressor) # Importante resumen estadístico para conocer sobre la regresion

# Predecir resultados con el conjunto de test

y_pred <- predict(regressor, newdata = trainingSet)

# Visualización de los resultados en el conjunto de entrenamiento

library(ggplot2)

ggplot() + 
        geom_point(aes(x = trainingSet$YearsExperience, 
                       y = trainingSet$Salary), colour =  "red") +
        geom_point(aes(x = testingSet$YearsExperience,
                       y = testingSet$Salary), colour = "pink") +
        geom_line(aes(x=trainingSet$YearsExperience, 
                      y = y_pred), colour = "blue") +
        ggtitle("Sueldo vs Años de Experiencia (Conjunto de entrenamiento)") +
        xlab("Años de Experiencia")+
        ylab("Sueldo en USD")