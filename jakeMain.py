import pickle
import random
import os
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

from deap import base, creator, tools


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Dataset choosing
print("Dataset:\n1: CIFAR-10\n2: CIFAR-100\n3: ImageNet")
datasetChoice = input(": ").strip()

dataDir = ""
class_count = 0
load_dotenv()
if datasetChoice == "1":
    dataDir = os.getenv("CIFAR10_DIR")
    class_count = 10
elif datasetChoice == "2":
    dataDir = os.getenv("CIFAR100_DIR")
    class_count = 100
elif datasetChoice == "3":
    dataDir = os.getenv("ImageNet_DIR")
    class_count = -1  # I DO NOT KNOW WHAT CLASS COUNT IMAGE NET HAS. PLEASE UPDATE THIS WHEN YOU KNOW!!!
else:
    exit(1)

# Load the images, and the labels.
data_dict = unpickle(dataDir)
train_images, train_labels = data_dict[b'data'], data_dict[b'labels']
train_images = train_images / 255.0
train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0  # resize
train_labels = to_categorical(train_labels, num_classes=class_count)  # one-hot encode the data

# ======= PREDATOR DEFINITION =======

# Define the model. This is arbitrary, please change if you know what you're doing.
predator = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(class_count, activation='softmax'),
])

# Compile the model
predator.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Initial training
subset_indices = [i for i in range(0, int(len(train_images) / 3.0))]  # Replace with the indices of the desired subset
subset_train_images = train_images[subset_indices]
subset_train_labels = train_labels[subset_indices]

predator.fit(subset_train_images, subset_train_labels, epochs=1,
             validation_data=(subset_train_images, subset_train_labels))


# ======= PREY DEFINITION =======
def evaluate_prey(individual):
    print(individual)
    indices = [round(i % len(train_images / 3.0)) for i in individual]
    predictions = predator.predict(train_images[indices], verbose=0)
    true_labels = tf.argmax(train_labels[indices], axis=1)
    predicted_labels = tf.argmax(predictions, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(true_labels, predicted_labels), dtype=tf.float32))

    # return predator.evaluate(train_images[indices], train_labels[indices])

    # print("Predicted : ", predicted_labels)
    # print("Data given: ", true_labels)
    # print("Accuracy  : ", accuracy.numpy())
    # input()
    return accuracy.numpy()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a toolbox with the required evolutionary operators
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, class_count)  # Min and max for output of prey
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute,
                 n=int(len(train_images) / 3.0))  # Change this to subset size. I've chosen 1/3rd arbitrarily.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_prey)

# Create an initial population
population = toolbox.population(n=5)

# Set the algorithm parameters
CXPB, MUTPB, NGEN = 0.7, 0.2, 3

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = (fit, )


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
            ind.fitness.values = (fit,)

        # Replace the old population by the offspring
        population[:] = offspring

    best_ind = tools.selBest(population, 1)[0]
    print(f"Best individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}")
    return best_ind


# ===============================


for global_rounds in range(10):
    best_individual = train_prey()

    indices = [round(i % len(train_images / 3.0)) for i in best_individual]

    # Train the model on the subset
    predator.fit(train_images[indices], train_labels[indices], epochs=10,
                 validation_data=(train_images[indices], train_labels[indices]))

test_loss, test_acc = predator.evaluate(train_images, train_labels)
print(f'Test accuracy: {test_acc}')
