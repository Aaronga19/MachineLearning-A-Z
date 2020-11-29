
        
#                                                       K-Means 

# Importan los datos
dataset = read.csv("GitHub/MachineLearningAZ/datasets/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_Customers.csv")
X = dataset[, 4:5]

# Método de codo
set.seed(6)
wcss = vector()
for (i in 1:10){
        wcss[i]<- sum(kmeans(X, centers = i)$withinss)
}
plot(1:10, wcss, type = "b", main = "Metodo del codo", xlab = "Número de Clusters (K)", ylab = "WCSS(K)")

# Aplicar el algoritmo de Kmeans con el k óptimo 
set.seed(29)
kmeans = kmeans(X,5, iter.max = 300, nstart = 10)

# Visualización de los Clusters 
library(cluster)
clusplot(X, kmeans$cluster, lines = 0, shade = TRUE, color = TRUE, labels = 4, plotchar = FALSE, span = TRUE,
         main = "Clustering de Clientes",
         xlab = "Ingresos anuales",
         ylab = "Puntuación (1-100)")
