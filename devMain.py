import random
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# Import EfficientNet (shown to have higher accuracy than ResNet50 with fewer parameters)
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import EfficientNetB7

from deap import base, creator, tools, algorithms

import threading
import psutil
import time

current_phase = 'none'

stop_logging = threading.Event()

try:
    import pynvml
    pynvml.nvmlInit()
except ImportError:
    print("pynvml not available, GPU logging will not be included.")

def continual_logging(file_path, interval=1, stop_event=None):
    global current_phase
    global logging_file_dir

    while os.path.isfile(file_path):
        file_path += ".log"

    logging_file_dir = file_path

    with open(file_path, 'w') as file:  # Open the file once and write headers
        file.write("timestamp, memory_usage_mb, cpu_usage_percent, gpu_usage_percent, phase\n")

    while not stop_event.is_set():
        # Memory usage in MB
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        # CPU usage in percent
        cpu_usage = psutil.cpu_percent(interval=None)

        log_message = f"{datetime.now()}, {memory_usage}, {cpu_usage}"

        # GPU usage in percent if enabled
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0 for the first GPU
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = gpu_util.gpu
            log_message += f", {gpu_usage}"
        except pynvml.NVMLError as e:
            log_message += ", N/A"
            print(e)

        # Add the phase to the log message
        log_message += f", {current_phase}\n"
        with open(file_path, 'a') as file:
            file.write(log_message)
        time.sleep(interval)

def load_cifar10():
    (train_images, train_labels), _ = tf.keras.datasets.cifar10.load_data()
    # train_images = train_images / 255.0
    # train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0  # resize
    # train_labels = to_categorical(train_labels, num_classes=10)
    return train_images, train_labels

def load_cifar100():
    (train_images, train_labels), _ = tf.keras.datasets.cifar100.load_data()
    # train_images = train_images / 255.0
    # train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0  # resize
    # train_labels = to_categorical(train_labels, num_classes=100)
    return train_images, train_labels

def log_indecies(indecies):
    # Log indecies for debug
    print(f"{str(datetime.now())} | Outputting indecies to log file")
    with open('indicesLog.txt', 'a') as file:
        file.write(', '.join(map(str, indecies)) + '\n')
    print(f"{str(datetime.now())} | Done")

# Clear log file
with open('indicesLog.txt', 'w'): 
    pass


# Default values. Can get overwritten in x.config
global_epochs = 30
prey_mini_epochs = 10
prey_partition_size = 0.25
predator_mini_epochs = 10
predator_start_epochs = 1
predator_batch_size = 64



print("Proceding with the following values:")
print(f"Global Epochs: {global_epochs}")
print(f"Prey Mini Epochs: {prey_mini_epochs}")
print(f"Prey Partition Size: {prey_partition_size}")
print(f"Predator Mini Epochs: {predator_mini_epochs}")
print(f"Predator Start Epochs: {predator_start_epochs}")
print(f"Predator Batch Size: {predator_batch_size}")

load_dotenv()

# Dataset choosing
train_images, train_labels = load_cifar100()
class_count = 100

train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # resize

train_labels = tf.one_hot(train_labels,
                     depth=np.array(train_labels).max() + 1,
                     dtype=tf.float64)
train_labels = tf.squeeze(train_labels)
train_labels = train_labels.numpy()
train_images, train_labels = shuffle(train_images, train_labels, random_state=1)

#train_labels = to_categorical(train_labels, num_classes=class_count)  # one-hot encode the data
# train_labels = tf.squeeze(train_labels)

starttime = datetime.now()
logging_thread = threading.Thread(target=continual_logging, args=('usage_log.txt', 1, stop_logging))
logging_thread.start()

# ======= PREDATOR DEFINITION =======

# # Model = ResNet50
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant
predator = Sequential()

predator.add(Conv2D(256,(3,3),padding='same',input_shape=(32,32,3)))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(Conv2D(256,(3,3),padding='same'))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(MaxPooling2D(pool_size=(2,2)))
predator.add(Dropout(0.2))

predator.add(Conv2D(512,(3,3),padding='same'))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(Conv2D(512,(3,3),padding='same'))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(MaxPooling2D(pool_size=(2,2)))
predator.add(Dropout(0.2))

predator.add(Conv2D(512,(3,3),padding='same'))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(Conv2D(512,(3,3),padding='same'))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(MaxPooling2D(pool_size=(2,2)))
predator.add(Dropout(0.2))

predator.add(Conv2D(512,(3,3),padding='same'))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(Conv2D(512,(3,3),padding='same'))
predator.add(BatchNormalization())
predator.add(Activation('relu'))
predator.add(MaxPooling2D(pool_size=(2,2)))
predator.add(Dropout(0.2))

predator.add(Flatten())
predator.add(Dense(1024))
predator.add(Activation('relu'))
predator.add(Dropout(0.2))
predator.add(BatchNormalization(momentum=0.95, 
        epsilon=0.005,
        beta_initializer=RandomNormal(mean=0.0, stddev=0.05), 
        gamma_initializer=Constant(value=0.9)))
predator.add(Dense(100,activation='softmax'))

predator.summary()

# Compile the model
predator.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

# Initial training
#subset_indices = [i for i in range(0, int(len(train_images) / 3.0))]  # Replace with the indices of the desired subset
#print(subset_indices)
#subset_train_images = train_images[subset_indices]
#subset_train_labels = train_labels[subset_indices]

predator.fit(
    train_images,
    train_labels,
    epochs = predator_start_epochs, # Increase epochs while training
    batch_size = predator_batch_size
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
    # input(
    
    return 1 - accuracy.numpy()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a toolbox with the required evolutionary operators
toolbox = base.Toolbox()
toolbox.register("attr_int", random.uniform, 0, len(train_images))  # Min and max for output of prey
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute,
                 n=int(len(train_images) * prey_partition_size))  # Change this to subset size. I've chosen 1/3rd arbitrarily.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1400, indpb=1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_prey)

# Create an initial population
population = toolbox.population(n=400)

# Set the algorithm parameters
CXPB, MUTPB, NGEN = 0.6, 0.4, prey_mini_epochs

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

from keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='accuracy', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('\n Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True

print("\n\nCurrently trying to fit 1/5th all data randomly got")
for global_rounds in range(global_epochs):
    current_round = global_rounds
    print(f"\n{str(datetime.now())} | Round {global_rounds + 1} Begin.")

    # Actual
    current_phase = 'prey'
    print(f"{str(datetime.now())} | Training prey...")
    best_individual = train_prey()

    indices = [round(i) % len(train_images) for i in best_individual]

    print(f"{str(datetime.now())} | Training predator with early stopping...")
    # log_indecies(indices)
    current_phase = 'predator'

    # Training predator with early stopping callback
    callbacks = [TerminateOnBaseline(monitor="accuracy",baseline=1.0)]

    # Train the model on the subset with early stopping
    predator.fit(train_images[indices], train_labels[indices], epochs = predator_mini_epochs, verbose=1, callbacks=callbacks, batch_size=predator_batch_size)

    # Predict
    print(f"{str(datetime.now())} | Making predictions...")
    predator_predictions = predator.predict(train_images, verbose=0)
    # full_loss, full_acc = predator.evaluate(train_images, train_labels)
    # print(f"{str(datetime.now())} | Train accuracy: {full_acc}")

    print(f"{str(datetime.now())} | Done!")


full_loss, full_acc = predator.evaluate(train_images, train_labels)

endtime = datetime.now()
stop_logging.set()
logging_thread.join()

print(f"{str(datetime.now())} | Train accuracy: {full_acc}")
print(f"That took {str(endtime - starttime)}")

# Append the logs that have been generated to the end of the log file
import loggymclogface

timestamps, memory_usage, cpu_usage, gpu_usage, phases = loggymclogface.read_usage_log(logging_file_dir)
loggymclogface.plot_usage_graphs(timestamps, memory_usage, cpu_usage, gpu_usage, phases)