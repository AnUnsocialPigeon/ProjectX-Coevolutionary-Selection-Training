import random
import time
import os
from dotenv import load_dotenv

from deap import base, creator, tools

# Predator libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from neuralNetwork import neuralNetwork
from generateData import generateData
from geneticAlgorithm import geneticAlgorithm

def main():

    #raise NotImplementedError("This is a work in progress - idk why we do some things")

    load_dotenv()

    FOLDER_PATH = os.getenv("FOLDER_PATH")

    predator = neuralNetwork()
    prey = geneticAlgorithm(FOLDER_PATH)

    for epoc in range(5):
        # Train the predator
        start_time = time.time()
        train_predators(predator, given, expected)

        # Test the predator
        predictions = predator.predict(given)
        fitness_value = predator_evaluation(predictions, expected)

        # Train the prey
        train_prey()
        prey_end_time = time.time()

        fitness_value = predator_evaluation(predictions.round(), expected) #why do we calculate fitness again?
        predator_end_time = time.time()

        print(f"Epoch time (Prey): {prey_end_time - start_time} seconds")
        print(f"Epoch time (Predator): {predator_end_time - prey_end_time} seconds")
        print(f"Epoch time (Combined): {predator_end_time - start_time} seconds")
        print(f"Predator Fitness: {fitness_value}")
        print(f"Predator Predicted: {predictions}")
        print(f"Predator Expected: {expected}")
        print("")

if __name__ == "__main__":
    main()
