from generateData import generateData


from functools import reduce

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
    
    def trainModel(self, chromosomes): #will have to change at integration stage
        # concatenate chromosomes
        # train nn in batches with custom .fit that gives us missclassifications for each gene
        # change genes in generateData
        filePaths = self.data.getFilePaths(reduce(lambda a, b: a+b, chromosomes))
        images, labels = np.array([self.data.getImageAndLabel(filePath) for filePath in filePaths]).T #do this in batches
        self.model.fit(images, labels, epochs=1, verbose=2)

    def modelEvaluation(self, chromosome):
        raise NotImplementedError("there is some code here, but it is wrong")
        filePaths = self.data.getData(chromosome)
        data = [self.data.getImageAndLabel(filePath) for filePath in filePaths]
        images = [i for i, j in data]
        labels = [j for i, j in data]

        correct_predictions = np.sum(self.model.predict(images).round() == labels) # this line requires processing - I have just put stuff on
        fitness_value = correct_predictions / len(chromosome)
        return fitness_value
    
    def chromosomeEvaluation(self, chromosomes):
        # just to simplify code in geneticAlgorithm.py
        return self.data.chromosomeEvaluation(chromosomes)
    