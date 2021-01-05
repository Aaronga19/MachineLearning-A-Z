
#                                       Reder Neuronales Artificiales

# Se carga el conjunto de datos

dataset = read.csv("GitHub/MachineLearningAZ/datasets/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv")

# Filtrado de columnas si es necesario
dataset = dataset[, 4:14]



# codificar variables categoricas para RNA

dataset$Geography = as.numeric(factor(dataset$Geography,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1,2,3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                           levels = c('Female','Male'),
                           labels = c(1,2)))


# Dividir los datos en conjunto de entrenamiento y conjunto de test

library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.80)
split
trainingSet = subset(dataset, split == T)
testingSet = subset(dataset, split == F)


# Escalado de valores 

trainingSet[,-11] = scale(trainingSet[,-11])
testingSet[,1:10] = scale(testingSet[,1:10])


# Crear la Red Neuronal 

#install.packages("h2o")
library(h2o)
classifier = h2o.deeplearning(y = "Exited",
                              training_frame = as.h2o(trainingSet),
                              activation = "Rectifier",
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2)



# Prediccion de nuevos resultados con el conjunto de Testing
prob_pred = h2o.predict(classifier, newdata = as.h2o(testingSet[,-11]))
y_pred = ifelse(prob_pred>0.5,1,0)
y_pred = (prob_pred>0.5)
y_pred = as.vector(y_pred)

# Crear matriz de Confusión 

cm = table(testingSet[,11], y_pred)

# Cerrar el servidor H2O

h2o.shutdown()