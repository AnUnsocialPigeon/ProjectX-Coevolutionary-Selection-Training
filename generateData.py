import random
import time
import os

from deap import base, creator, tools

# Predator libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# def generate_data(num_samples, array_size):
#     given = np.random.choice(['A', 'B', 'C'], size=(num_samples, array_size))
#     expected = (given == 'B').astype(int) + (2 * (given == 'A').astype(int))
#     return given, expected

def increment(i=0):
    while True:
        yield i
        i += 1

class generateData():

    def __init__(self, folderPath):
        self.folderPath = folderPath
        self.geneToFilePath = {i:os.join(self.filePath, j) for i, j in zip(increment(), os.listdir(self.folderPath))}
        self.filePathToGene = {j:i for i, j in self.geneToFilePath.items()}
        
    def getData(self, chromosome):
        return [self.geneToFilePath[i] for i in chromosome]
    
    def getStructure(self):
        return len(self.geneToFilePath)
    
    def getImageAndLabel(self, filePath): #isn't generalised to all types of data
        with open(filePath, 'rb') as f:
            data = f.read()

        return (data, filePath.split('\\')[-1])
    
    def getImage(self, filePath):
        with open(filePath, 'rb') as f:
            data = f.read()

        return data

    def getChromosome(self, filePaths):
        n = self.getStructure()
        chromosome = [0]*n
        for filePath in filePaths:
            chromosome[self.filePathToGene[filePath]] = 1

        return chromosome

    def geneEvaluation(self, chromosome):
        '''
            implement the forgetfullness function here
        '''

        raise NotImplementedError("geneEvaluation is not implemented yet")
