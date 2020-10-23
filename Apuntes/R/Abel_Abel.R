print("Y me llamo Abelito")

dataset = read.csv("GitHub/MachineLearningAZ/Apuntes/R/Ejercicio_4Abel.csv")

library(dplyr)
dataset = select(dataset, -X)

dataset = dataset[-c(107,108,109), ]
dataset = select(dataset, -ï..Trimestre)
names(dataset)

library(caTools)
set.seed(123)
split = sample.split(dataset$Ms, SplitRatio = 0.8)
trainingSet = subset(dataset, split == T)
testingSet = subset(dataset, split == F)

regression = lm(formula = Ms~ PIB, data = dataset) # "." Es para indicar todas las variables IN-dependientes
summary(regression)

y_pred = predict(regression, newdata = testingSet)

library(ggplot2)

ggplot() + 
        geom_point(aes(x = dataset$PIB, 
                       y = dataset$Ms), colour =  "red") +
        geom_line(aes(x=testingSet$PIB, 
                      y = y_pred), colour = "blue") +
        ggtitle("Ms vs PIB (Conjunto de entrenamiento)") +
        xlab("PIB")+
        ylab("Importaciones")