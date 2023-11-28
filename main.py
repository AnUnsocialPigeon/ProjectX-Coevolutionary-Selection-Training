import random
import time
from deap import base, creator, tools

# Predator libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


def prey_evaluation(individual):
    # Current task - get this function to evaluate the pray as according to the predator
    return 1

def predator_evaluation(predator_output, expected):
    correct_predictions = np.sum(predator_output.round() == expected)
    fitness_value = correct_predictions / len(expected)
    return fitness_value

# Define the individual and population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a toolbox with the required evolutionary operators
toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", prey_evaluation)

# ============== PREY ==============
# Create an initial population
population = toolbox.population(n=70)

# Set the algorithm parameters
CXPB, MUTPB, NGEN = 0.7, 0.2, 50

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit


def train_prey():
    # Begin the evolution
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the old population by the offspring
        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}")


# ============== PREDATOR ==============
# Define the model globally
predator = Sequential()
predator.add(Dense(units=4, input_dim=10, activation='relu'))
predator.add(Dense(units=1, activation='sigmoid'))
predator.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

num_samples = 5
array_size = 10


def generate_data(num_samples, array_size):
    given = np.random.choice(['A', 'B', 'C'], size=(num_samples, array_size))
    expected = (given == 'B').astype(int) + (2 * (given == 'A').astype(int))
    return given, expected


def train_predators(model, given, expected_output, epochs=4):
    model.fit(given, expected_output, epochs=epochs, verbose=2)


if __name__ == "__main__":
    given, expected = generate_data(num_samples, array_size)
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

        fitness_value = predator_evaluation(predictions.round(), )
        predator_end_time = time.time()

        print(f"Epoch time (Prey): {prey_end_time - start_time} seconds")
        print(f"Epoch time (Predator): {predator_end_time - prey_end_time} seconds")
        print(f"Epoch time (Combined): {predator_end_time - start_time} seconds")

