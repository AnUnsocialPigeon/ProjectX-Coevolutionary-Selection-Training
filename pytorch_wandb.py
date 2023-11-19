import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import wandb

# Initialize W&B
wandb.init(project="deap-evolutionary-algorithm")

# Define the problem: maximize the sum of a list of numbers
def evaluate(individual):
    return sum(individual),

# Define the individual and population
class Individual(nn.Module):
    def __init__(self, n_genes):
        super(Individual, self).__init__()
        self.genes = nn.Parameter(torch.rand(n_genes))

# Create a toolbox with the required evolutionary operators
def individual():
    return Individual(n_genes=10)

def population(n):
    return [individual() for _ in range(n)]

def mate(ind1, ind2):
    # Two-point crossover
    crossover_points = sorted(random.sample(range(len(ind1.genes)), 2))
    ind1.genes[crossover_points[0]:crossover_points[1]], ind2.genes[crossover_points[0]:crossover_points[1]] = \
        ind2.genes[crossover_points[0]:crossover_points[1]].clone(), ind1.genes[crossover_points[0]:crossover_points[1]].clone()

def mutate(ind, mu=0, sigma=1, indpb=0.2):
    # Gaussian mutation
    mask = torch.rand_like(ind.genes) < indpb
    ind.genes += mask * torch.normal(mu, sigma, size=ind.genes.shape)

def select(population, k, tournsize=3):
    # Tournament selection
    chosen = []
    for _ in range(k):
        aspirants = random.sample(population, tournsize)
        chosen.append(max(aspirants, key=lambda ind: sum(ind.genes)))
    return chosen

def evaluate_population(population):
    return [evaluate(ind.genes) for ind in population]

def main():
    # Create an initial population
    population_size = 50
    population = population(population_size)

    # Set up W&B logging
    wandb.watch(population, log="all")

    # Set the algorithm parameters
    CXPB, MUTPB, NGEN = 0.7, 0.2, 40

    # Evaluate the entire population
    fitnesses = evaluate_population(population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness = fit

    # Begin the evolution
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = select(population, len(population))

        # Clone the selected individuals
        offspring = [torch.clone(ind) for ind in offspring]

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                mate(child1, child2)

        for mutant in offspring:
            if random.random() < MUTPB:
                mutate(mutant)

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not hasattr(ind, 'fitness')]
        fitnesses = evaluate_population(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness = fit

        # Replace the old population by the offspring
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [sum(ind.fitness) for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        # Log the generation information to W&B
        wandb.log({"generation": gen, "min_fitness": min(fits), "max_fitness": max(fits), "avg_fitness": mean, "std_fitness": std})

    best_ind = max(population, key=lambda ind: sum(ind.fitness))
    print(f"Best individual: {best_ind.genes}, Fitness: {sum(best_ind.fitness)}")

if __name__ == "__main__":
    main()
