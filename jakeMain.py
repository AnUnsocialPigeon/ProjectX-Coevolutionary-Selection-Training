import os
import sys
import pickle
import random
from datetime import datetime
from dotenv import load_dotenv

import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical

from deap import base, creator, tools


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Load the images, and the labels.
data_dict = unpickle(dataDir)
train_images, train_labels = data_dict[b'data'], data_dict[b'labels']
train_images = train_images / 255.0
train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0  # resize
train_labels = to_categorical(train_labels, num_classes=class_count)  # one-hot encode the data

starttime = datetime.now()

# Initial training
subset_indices = [i for i in range(0, int(len(train_images) / 3.0))]  # Replace with the indices of the desired subset
subset_train_images = train_images[subset_indices]
subset_train_labels = train_labels[subset_indices]

predator.fit(subset_train_images, subset_train_labels, epochs=1,
             validation_data=(subset_train_images, subset_train_labels))

predator_predictions = predator.predict(train_images, verbose=0)


# ======= PREY DEFINITION =======
def evaluate_prey(individual):
    #print(individual)
    indices = [round(i) % len(train_images) for i in individual]
    predictions = predator_predictions[indices]
    true_labels = tf.argmax(train_labels[indices], axis=1)
    predicted_labels = tf.argmax(predictions, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(true_labels, predicted_labels), dtype=tf.float32))

    # return predator.evaluate(train_images[indices], train_labels[indices])

    # print("Predicted : ", predicted_labels)
    # print("Data given: ", true_labels)
    # print("Accuracy  : ", accuracy.numpy())
    # input()
    return 1 - accuracy.numpy()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a toolbox with the required evolutionary operators
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, class_count)  # Min and max for output of prey
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute,
                 n=int(len(train_images) / 5.0))  # Change this to subset size. I've chosen 1/3rd arbitrarily.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_prey)

# Create an initial population
population = toolbox.population(n=70)

# Set the algorithm parameters
CXPB, MUTPB, NGEN = 0.7, 0.2, 50

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = (fit, )


# ===============================

for global_rounds in range(10):
    print(f"\n{str(datetime.now())} | Round {global_rounds + 1} Begin.")
    print(f"{str(datetime.now())} | Training prey...")
    best_individual = train_prey()

    indices = [round(i) % len(train_images) for i in best_individual]

    print(f"{str(datetime.now())} | Training predator...")
    # Train the model on the subset
    predator.fit(train_images[indices], train_labels[indices], epochs=10,
                validation_data=(train_images[indices], train_labels[indices]), verbose=0)
    
    print(f"{str(datetime.now())} | Making predictions...")
    predator_predictions = predator.predict(train_images, verbose=0)
    print(f"{str(datetime.now())} | Done!")


test_loss, test_acc = predator.evaluate(train_images, train_labels)

endtime = datetime.now()

print(f"{str(datetime.now())} | Test accuracy: {test_acc}")
print(f"That took {str(endtime - starttime)}")
