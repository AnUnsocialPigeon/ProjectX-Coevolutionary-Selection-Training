import random
import pickle
from datetime import datetime
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50 # Good image model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from efficientnet.tfkeras import EfficientNetB0

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

# Default values. Can get overwritten in x.config
global_epochs = 10
prey_mini_epochs = 5
prey_partition_size = 0.2
predator_mini_epochs = 5
predator_start_epochs = 1
predator_batch_size = 32

# Obtaining values from config file
try:
    with open('x.config') as file:
        for line in file:
            try:
                parts = line.split(':')
                if parts[0] == "Global Epochs":
                    global_epochs = int(parts[1])
                elif parts[0] == "Prey Mini Epochs":
                    prey_mini_epochs = int(parts[1])
                elif parts[0] == "Predator Mini Epochs":
                    predator_mini_epochs = int(parts[1])
                elif parts[0] == "Predator Start Epochs":
                    predator_start_epochs = int(parts[1])
                elif parts[0] == "Prey Partition Size":
                    prey_partition_size = float(parts[1])
                elif parts[0] == "Predator Batch Size":
                    predator_batch_size = int(parts[1])
                else:
                    print("Unrecognised config line: " + str(line))                    
            except Exception as e:
                print(e)
except FileNotFoundError as e:
    print(e)


print("Proceding with the following values:")
print(f"Global Epochs: {global_epochs}")
print(f"Prey Mini Epochs: {prey_mini_epochs}")
print(f"Prey Partition Size: {prey_partition_size}")
print(f"Predator Mini Epochs: {predator_mini_epochs}")
print(f"Predator Start Epochs: {predator_start_epochs}")
print(f"Predator Batch Size: {predator_batch_size}")

class_count = 100

data_dict = unpickle(os.getenv("CIFAR100_DIR"))
train_images, train_labels = data_dict[b'data'], data_dict[b'fine_labels']
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

# ======= PREDATOR DEFINITION =======

# # Model = ResNet50 or EfficientNetB0
predator = ResNet50(
# predator = EfficientNetB0(
    weights = None,
    input_shape = (32,32,3),
    classes = class_count
)

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
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True

for global_rounds in range(global_epochs):
    print(f"\n{str(datetime.now())} | Round {global_rounds + 1} Begin.")

    # Training predator with early stopping callback
    callbacks = [TerminateOnBaseline(monitor="accuracy",baseline=1.0)]

    # Train the model on the subset with early stopping
    predator.fit(train_images, train_labels, epochs = predator_mini_epochs, verbose=1, callbacks=callbacks, batch_size=predator_batch_size)


full_loss, full_acc = predator.evaluate(train_images, train_labels)

endtime = datetime.now()

print(f"{str(datetime.now())} | Train accuracy: {full_acc}")
print(f"That took {str(endtime - starttime)}")