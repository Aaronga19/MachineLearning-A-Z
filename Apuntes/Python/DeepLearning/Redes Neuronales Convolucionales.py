# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:42:49 2021

@author: Aaronga
"""

"""                                 Redes Neuronales Convolucionales                                """

# Conda install -c conda-forge keras
# Instalar Theano


# Construir modelos de CNN
# Importar librerías y paquetes
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Inicializar la CNN
classifier = Sequential() # Capas (Dendritas)

# Paso 1 - Convolución.
classifier.add(Conv2D(filters= 32,kernel_size=(3,3), 
                      input_shape = (64,64,3), activation="relu"))
# Paso 2 - Max Pooling.
classifier.add(MaxPooling2D(pool_size=(2,2)))
            # Una segunda capa de convolución y Maxpool (Para contemplar detalles)
classifier.add(Conv2D(filters= 32,kernel_size=(3,3), 
                      input_shape = (64,64,3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Paso 3 - Flattering.
classifier.add(Flatten())

# Paso 4 - Full Conection
classifier.add(Dense(units=128, activation = "relu"))
classifier.add(Dense(units= 1, activation = "sigmoid"))
"""Podemos añadir más redes o capas para mejorar el algorítmo"""

# Compilar la CNN
classifier.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

# Ajustar la CNN a las imágenes para entrenar 
 
from keras.preprocessing.image import ImageDataGenerator

trainDataGen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

testDataGen = ImageDataGenerator(rescale=1./255)

trainingData = trainDataGen.flow_from_directory('C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set',
                                                  target_size= (64,64),
                                                  batch_size = 32,
                                                  class_mode = 'binary')

testData = testDataGen.flow_from_directory('C:/Users/Aaronga/Documents/GitHub/MachineLearningAZ/datasets/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set',
                                                  target_size= (64,64),
                                                  batch_size = 32,
                                                  class_mode = 'binary')

classifier.fit_generator(trainingData,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data= testData,
                    validation_steps=2000)
 







