from generateData import generateData

# Predator libraries
import tensorflow as tf
from keras import layers, models
from keras.layers import Dense
import numpy as np


'''
generate data is called data in this class, idk why I chose to do that, but oh well.
'''

class neuralNetwork():
    def __init__(self, folder_path, class_count):
        
        self.data = generateData(folder_path)
        self.getModel()
        self.class_count = class_count

    def getModel(self): #will have to change at integration stage

        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.class_count, activation='softmax'),
        ])

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def queryModel(self, chromosome): #will have to change at integration stage
        filePaths = self.data.getFilePaths(chromosome)
        data = np.array([self.data.getImage(filePath) for filePath in filePaths])
        return self.model.predict(data)
    
    def trainModel(self, chromosomes, epoch=4): #will have to change at integration stage
        filePaths = self.data.getFilePaths(chromosomes)
        images, labels = np.array([self.data.getImageAndLabel(filePath) for filePath in filePaths]).T
        self.model.fit(images, labels, epochs=epoch, verbose=2)
        # return chromosome evaluation metrics

    def modelEvaluation(self, chromosome):
        filePaths = self.data.getData(chromosome)
        data = [self.data.getImageAndLabel(filePath) for filePath in filePaths]
        images = [i for i, j in data]
        labels = [j for i, j in data]
        
        

        correct_predictions = np.sum(self.model.predict(images).round() == labels) # this line requires processing - I have just put stuff on
        fitness_value = correct_predictions / len(chromosome)
        return fitness_value
    
    def chromosomeEvaluation(self, chromosomes, missclassifications, number):
        # just to simplify code in geneticAlgorithm.py
        return self.data.chromosomeEvaluation(chromosomes, missclassifications, number)
    