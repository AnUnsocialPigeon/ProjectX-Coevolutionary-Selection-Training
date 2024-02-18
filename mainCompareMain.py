import random
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms
import time

time_taken_without_cest = []

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
global_epochs = 5
prey_mini_epochs = 5
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

# Dataset choosing
# print("Dataset:\n1: CIFAR-10\n2: CIFAR-100\n3: ImageNet")
datasetChoice = "2"

load_dotenv()

if datasetChoice == "1":
    data_dict = os.getenv("CIFAR10_DIR")
    train_images, train_labels = load_cifar10()
    class_count = 10
elif datasetChoice == "2":
    data_dict = os.getenv("CIFAR100_DIR")
    train_images, train_labels = load_cifar100()
    class_count = 100
elif datasetChoice == "3":
    data_dict = os.getenv("ImageNet_DIR")
    # Load ImageNet dataset
else:
    exit(1)

class_count = 100

train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # resize

train_labels = tf.one_hot(train_labels,
                     depth=np.array(train_labels).max() + 1,
                     dtype=tf.float64)
train_labels = tf.squeeze(train_labels)
train_labels = train_labels.numpy()
train_images, train_labels = shuffle(train_images, train_labels, random_state=1)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=1)


#train_labels = to_categorical(train_labels, num_classes=class_count)  # one-hot encode the data
# train_labels = tf.squeeze(train_labels)

starttime = datetime.now()

# ======= PREDATOR DEFINITION =======

# # Model = 
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

## Uncomment for testing
# # Compile the model
# predator.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4),
#                  loss=tf.keras.losses.CategoricalCrossentropy(),
#                  metrics=['accuracy'])

# Initial training
#subset_indices = [i for i in range(0, int(len(train_images) / 3.0))]  # Replace with the indices of the desired subset
#print(subset_indices)
#subset_train_images = train_images[subset_indices]
#subset_train_labels = train_labels[subset_indices]

predator_predictions = predator.predict(train_images, verbose=0)
print("Predator created...")

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
    print(f"\n{str(datetime.now())} | Round {global_rounds + 1} Begin.")
    # Time before training
    start_time = time.time()

    print(f"{str(datetime.now())} | Training predator with early stopping...")

    # Training predator with early stopping callback
    callbacks = [TerminateOnBaseline(monitor="accuracy",baseline=1.0)]

    # Train the model on the subset with early stopping
    predator.fit(train_images, train_labels, epochs = predator_mini_epochs, verbose=1, callbacks=callbacks, batch_size=predator_batch_size)

    # Predict
    print(f"{str(datetime.now())} | Making predictions...")
    # predator_predictions = predator.predict(train_images, verbose=0)
    # full_loss, full_acc = predator.evaluate(train_images, train_labels)
    # print(f"{str(datetime.now())} | Train accuracy: {full_acc}")
    # Time after training
    end_time = time.time()

    # Calculate time taken
    time_taken = end_time - start_time

    time_taken_without_cest.append(time_taken)

    print(f"{str(datetime.now())} | Done!")


full_loss, full_acc = predator.evaluate(train_images, train_labels)

endtime = datetime.now()

print(f"{str(datetime.now())} | Train accuracy: {full_acc}")
print(f"That took {str(endtime - starttime)}")

epochs = range(1, global_epochs + 1)

plt.plot(epochs, time_taken_without_cest, label='Without CEST')
plt.xlabel('Epochs')
plt.ylabel('Time taken (seconds)')
plt.title('Time taken vs Epochs')
plt.legend()
plt.grid(True)
plt.show()
