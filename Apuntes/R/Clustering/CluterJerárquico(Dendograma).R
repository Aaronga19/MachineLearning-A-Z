

#                                               Clustering Jerárquico

# Importar dataset 
dataset <- read.csv("GitHub/MachineLearningAZ/datasets/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Mall_Customers.csv")
X <- dataset[, 4:5]

# Utilizar el dendograma para encontrar el número óptimo de clusters 
dendogram <- hclust(dist(X, method = "euclidean"),
                    method = "ward.D")
# Visualizar Dendograma
plot(dendogram,
     main = "Dendograma",
     xlab = "Clientes",
     ylab = "Distancia Euclidea") # En este caso salen 5 clusters

# Ajustar Clustering Jerárquico a nuestro dataset 

hc <- hclust(dist(X, method = "euclidean"),
                    method = "ward.D")
y_hc <- cutree(hc, k= 5)

# Visualización de los cluster 

library(cluster)
clusplot(X, y_hc, lines = 0, shade = TRUE, color = TRUE, labels = 4, plotchar = FALSE, span = TRUE,
         main = "Clustering de Clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuación (1-100)")