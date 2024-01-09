import random
import pickle
from datetime import datetime
import os
from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50 # Good image model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from deap import base, creator, tools


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def log_indecies(indecies):
    # Log indecies for debug
    print(f"{str(datetime.now())} | Outputting indecies to log file")
    with open('indicesLog.txt', 'a') as file:
        file.write(', '.join(map(str, indecies)) + '\n')
    print(f"{str(datetime.now())} | Done")

# Clear log file
with open('indicesLog.txt', 'w'): 
    pass


# Dataset choosing
print("Dataset:\n1: CIFAR-10\n2: CIFAR-100\n3: ImageNet")
datasetChoice = input(": ").strip()

load_dotenv()
if datasetChoice == "1":
    dataDir = os.getenv("CIFAR10_DIR")
    class_count = 10
elif datasetChoice == "2":
    dataDir = os.getenv("CIFAR100_DIR")
    class_count = 100
elif datasetChoice == "3":
    dataDir = os.getenv("ImageNet_DIR")
    # Load ImageNet dataset
else:
    exit(1)

data_dict = unpickle(dataDir)
train_images, train_labels = data_dict[b'data'], data_dict[b'labels']
train_images = train_images / 255.0
train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0  # resize
train_labels = to_categorical(train_labels, num_classes=class_count)  # one-hot encode the data
# train_labels = tf.squeeze(train_labels)

starttime = datetime.now()

# ======= PREDATOR DEFINITION =======

# Define the model. This is arbitrary, please change if you know what you're doing.

# # Model = Sequential. 
# predator = models.Sequential([
#     layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.BatchNormalization(),
#     #layers.Dropout(0.5),
#     layers.Dense(class_count, activation='softmax'),
# ])

# # Model = ResNet50
predator = ResNet50(
    weights = None,
    input_shape = (32,32,3),
    classes = class_count
)

predator.summary()

# Compile the model
predator.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Initial training
subset_indices = [i for i in range(0, int(len(train_images) / 3.0))]  # Replace with the indices of the desired subset
subset_train_images = train_images[subset_indices]
subset_train_labels = train_labels[subset_indices]

max_epochs_predator = 10
predator.fit(
    subset_train_images, 
    subset_train_labels, 
    epochs = 1, # Increase epochs while training
    batch_size = 64,
    validation_data = (subset_train_images, subset_train_labels)
)

predator_predictions = predator.predict(train_images, verbose=0)
print("Predator created...")

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
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1400, indpb=1)    
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_prey)

# Create an initial population
population = toolbox.population(n=70)

# Set the algorithm parameters
CXPB, MUTPB, NGEN = 0.7, 0.2, 5

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
    print(f"{str(datetime.now())} | Best individual Fitness: {best_ind.fitness.values[0]}")
    return best_ind


# ===============================

print("\n\nCurrently trying to fit 1/5th all data randomly got")
for global_rounds in range(3):
    print(f"\n{str(datetime.now())} | Round {global_rounds + 1} Begin.")

    # Test
    #print(f"{str(datetime.now())} | Training predator...")
    # All
    #predator.fit(train_images, train_labels, epochs=10, validation_data=(train_images, train_labels), verbose=0)
    
    # Selection 
    #indecies = [i for i in range(0, int(len(train_images / 5)))]        # First selection
    # indecies = [random.randint(0, len(train_images) - 1) for i in range(int(len(train_images) / 5))] # Random Section
    # predator.fit(train_images[indecies], train_labels[indecies], epochs=10, validation_data=(train_images[indecies], train_labels[indecies]), verbose=0)


    # log_indecies(indecies)    
    # continue
    
    # Actual
    print(f"{str(datetime.now())} | Training prey...")
    best_individual = train_prey()


    indices = [round(i) % len(train_images) for i in best_individual]

    print(f"{str(datetime.now())} | Training predator with early stopping...")
    log_indecies(indices)

    # Training predator with early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Can vary patience

    # Train the model on the subset with early stopping
    predator.fit(train_images[indices], train_labels[indices], epochs = max_epochs_predator,
                validation_data=(train_images[indices], train_labels[indices]),
                callbacks=[early_stopping], verbose=1)
    
    # Log indecies for debug
    print(f"{str(datetime.now())} | Outputting indecies to log file")
    with open('indicesLog.txt', 'a') as file:
        file.write(', '.join(map(str, indices)) + '\n')

    # Predict
    print(f"{str(datetime.now())} | Making predictions...")
    predator_predictions = predator.predict(train_images, verbose=0)

    print(f"{str(datetime.now())} | Done!")


test_loss, test_acc = predator.evaluate(train_images, train_labels)

endtime = datetime.now()

print(f"{str(datetime.now())} | Test accuracy: {test_acc}")
print(f"That took {str(endtime - starttime)}")
