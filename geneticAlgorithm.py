import random
import time
from deap import base, creator, tools

# Predator libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from neuralNetwork import neuralNetwork
from generateData import generateData

class geneticAlgorithm():

    def __init__(self, folderPath):
        self.neuralNetwork = neuralNetwork(generateData(folderPath))
        self.chromosomeLength, self.numberOfClasses = self.neuralNetwork.data.getStructure() 

        # Define the individual and population
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create a toolbox with the required evolutionary operators
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.random) # individuals are encoded as binary vectors so it should be either 1 or 0
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.chromosomeLength) # this creates individual of 10 genes
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.neuralNetwork.chromosomeEvaluation())

        # ============== PREY ==============
        # Create an initial population
        self.population = self.toolbox.population(n=70) # population of 70 individuals 

        # Set the algorithm parameters
        self.CXPB, self.MUTPB, self.NGEN = 0.7, 0.2, 50

        # Evaluate the entire population - what are we evaluation when we haven't done any training yet? (copied from Jake's old Main)
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            # Remove brackets when the old prey function has been sorted, and new geneEvaluation function has been implemented
            ind.fitness.values = (fit,) 
            
    
    def train_prey(self):
        # Begin the evolution
        for gen in range(self.NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the old population by the offspring
            self.population[:] = offspring

        best_ind = tools.selBest(self.population, 1)[0]
        print(f"Best individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}")