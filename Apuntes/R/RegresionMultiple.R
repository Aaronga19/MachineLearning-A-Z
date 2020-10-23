
# Regresion Lineal Multiple
startUps = read.csv("C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_startups.csv")

#dataset = dataset[,2:3]

# codificar variables categoricas


startUps$State = factor(startUps$State,
                        levels = c("New York", "California", "Florida"),
                        labels = c(1,2,3))

# Dividir los datos en conjunto de entrenamiento y conjunto de test

library(caTools)
set.seed(123)
split = sample.split(startUps$Profit, SplitRatio = 0.8)
trainingSet = subset(startUps, split == T)
testingSet = subset(startUps, split == F)

# Ajustar el modelo de regresión lineal multiple con el conjunto de entrenamiento (Training_set)
regresion = lm(formula = Profit~., data = trainingSet) # "." Es para indicar todas las variables IN-dependientes
summary(regresion)

# Predecir los datos con el conjunto de Testing.
y_pred = predict(regresion, newdata = testingSet)

# Construir un modelo optimo con la eliminacion hacia atras
"""
SL = 0.05

regresion = lm(formula = Profit~ R.D.Spend + Administration + Marketing.Spend + State, data = trainingSet) # "." Es para indicar todas las variables IN-dependientes
summary(regresion)

regresion = lm(formula = Profit~ R.D.Spend + Administration + Marketing.Spend, data = trainingSet) # "." Es para indicar todas las variables IN-dependientes
summary(regresion)

regresion = lm(formula = Profit~ R.D.Spend +  Marketing.Spend, data = trainingSet) # "." Es para indicar todas las variables IN-dependientes
summary(regresion)

regresion = lm(formula = Profit~ R.D.Spend, data = trainingSet) # "." Es para indicar todas las variables IN-dependientes
summary(regresion)
"""
backwardElimination <- function(x, sl) {
        numVars = length(x)
        for (i in c(1:numVars)){
                regressor = lm(formula = Profit ~ ., data = x)
                maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
                if (maxVar > sl){
                        j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
                        x = x[, -j]
                }
                numVars = numVars - 1
        }
        return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)

# Para visualizar los datos

library(ggplot2)

ggplot() + 
        geom_point(aes(x = trainingSet$R.D.Spend, 
                       y = trainingSet$Profit), colour =  "red") +
        geom_point(aes(x = testingSet$R.D.Spend,
                       y = testingSet$Profit), colour = "pink") +
        geom_line(aes(x=trainingSet$R.D.Spend, 
                      y = y_pred), colour = "blue") +
        ggtitle("Sueldo vs Años de Experiencia (Conjunto de entrenamiento)") +
        xlab("Años de Experiencia")+
        ylab("Sueldo en USD")