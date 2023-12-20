import os
import sys
import time
import random
from datetime import datetime
from dotenv import load_dotenv

from deap import base, creator, tools

# Predator libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from neuralNetwork import neuralNetwork
from generateData import generateData
from geneticAlgorithm import geneticAlgorithm

def main(FOLDER_PATH, CLASS_COUNT):

    EPOCHS = 5
    predator = neuralNetwork()
    prey = geneticAlgorithm(FOLDER_PATH, CLASS_COUNT) #shouldnt call this prey

    for _ in range(EPOCHS):
        
        # Train the predator
        prey.neuralNetwork.trainModel(prey.population)

        # Train the prey
        prey.train_prey()

        # TODO: Probably print timing info as well
        # TODO: probably add in relevant print statements in the various classes
        # TODO: ACTUAL neural network stuff such as validation

# TODO: evaluate the predator against the entire dataset to see how good it is


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <FOLDER PATH> <CLASS COUNT>")
        sys.exit(1)
    
    FOLDER_PATH = sys.argv[1]
    CLASS_COUNT = sys.argv[2] #could count classes ourselves

    main(FOLDER_PATH, CLASS_COUNT)
