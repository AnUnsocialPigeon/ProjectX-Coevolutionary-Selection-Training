import random
import time
from deap import base, creator, tools
from generateData import generateData


# Predator libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# def train_predators(model, given, expected_output, epochs=4):
#     model.fit(given, expected_output, epochs=epochs, verbose=2)

'''
generate data is called data in this class, idk why I chose to do that, but oh well.
'''

class neuralNetwork():
    def __init__(self, data=generateData()):
        
        self.data = data
        self.getModel()

    def getModel(self): #will have to change at integration stage
        self.model = Sequential()
        self.model.add(Dense(units=4, input_dim=10, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def queryModel(self, chromosome): #will have to change at integration stage
        filePaths = self.data.getData(chromosome)
        data = [self.data.getImage(filePath) for filePath in filePaths]
        return self.model.predict(data)
    
    def trainModel(self, chromosome, epoch=4): #will have to change at integration stage
        '''
            this should return incorrect predictions and anything else that can be useful for a fitness function
        '''

        filePaths = self.data.getData(chromosome)
        data = [self.data.getImageAndLabel(filePath) for filePath in filePaths]
        images = [i for i, j in data]
        labels = [j for i, j in data]
        self.model.fit(images, labels, epochs=epoch, verbose=2)

    def modelEvaluation(self, chromosome):
        
        raise NotImplementedError

        filePaths = self.data.getData(chromosome)
        data = [self.data.getImageAndLabel(filePath) for filePath in filePaths]
        images = [i for i, j in data]
        labels = [j for i, j in data]
        
        

        correct_predictions = np.sum(self.model.predict(images).round() == labels) # this line requires processing - I have just put stuff on
        fitness_value = correct_predictions / len(chromosome)
        return fitness_value