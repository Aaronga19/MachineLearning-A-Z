#                                       Natural Language Processing (Bag of Words)

# Importar el dataset
datasetOriginal = read.delim("GitHub/MachineLearningAZ/datasets/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv", 
                     quote = "", stringsAsFactors = F)


# Limpieza del texto
# install.packages("tm")
# install.packages("SnowballC")
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(datasetOriginal$Review))
corpus = tm_map(corpus, content_transformer(tolower))
# Para consultar un elemento del corpus
# as.character(corpus[[1]])
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords("en"))
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Crear modelo de Bag of Words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = datasetOriginal$Liked


'''  Ajustando modelo de Clasificación (Random Forest) '''

# Colocar la variable de clasificacion como factor ya que Bayes lo pide en la librería 

dataset$Liked = factor(dataset$Liked,
                           levels = c(0,1)) 

# Dividir los datos en conjunto de entrenamiento y conjunto de test

library(caTools)


