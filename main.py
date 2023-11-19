import random
from deap import base, creator, tools
import wandb

# Initialize W&B
wandb.init(project="deap-evolutionary-algorithm")

# Define the problem: maximize the sum of a list of numbers
def evaluate(individual):
    return sum(individual),

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
toolbox.register("evaluate", evaluate)

def main():
    # Create an initial population
    population = toolbox.population(n=50)

    # Set up W&B logging
    wandb.watch(population, log="all")

    # Set the algorithm parameters
    CXPB, MUTPB, NGEN = 0.7, 0.2, 40

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

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

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        # Log the generation information to W&B
        wandb.log({"generation": gen, "min_fitness": min(fits), "max_fitness": max(fits), "avg_fitness": mean, "std_fitness": std})

    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}")

if __name__ == "__main__":
    main()
