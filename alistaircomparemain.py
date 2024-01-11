import random
import pickle
from datetime import datetime
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50 # Good image model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
#from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras import optimizers

from deap import base, creator, tools

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
        
        print("MEMORY USAGE: " + str(memory_usage))
        
        # CPU usage in percent
        cpu_usage = psutil.cpu_percent(interval=None)

        print("CPU USAGE: " + str(cpu_usage))
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
global_epochs = 30
predator_mini_epochs = 1
predator_start_epochs = 1
predator_batch_size = 64


print("Proceding with the following values:")
print(f"Global Epochs: {global_epochs}")
print(f"Predator Mini Epochs: {predator_mini_epochs}")
print(f"Predator Start Epochs: {predator_start_epochs}")
print(f"Predator Batch Size: {predator_batch_size}")

class_count = 100

data_dict = unpickle(r'C:/Users/jdtur/Downloads/cifar-100-python/cifar-100-python/train')
train_images, train_labels = data_dict[b'data'], data_dict[b'fine_labels']
train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # resize

train_labels = tf.one_hot(train_labels,
                     depth=np.array(train_labels).max() + 1,
                     dtype=tf.float64)
train_labels = tf.squeeze(train_labels)
train_labels = train_labels.numpy()
train_images, train_labels = shuffle(train_images, train_labels, random_state=1)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=1)

starttime = datetime.now()
logging_thread = threading.Thread(target=continual_logging, args=('usage_log.txt', 1, stop_logging))
logging_thread.start()

# ======= PREDATOR DEFINITION =======

# Batch norm model 4
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
predator.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4),
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy'])

print("Predator created...")

# ===============================

for global_rounds in range(global_epochs):
    current_round = global_rounds
    current_phase = 'predator'
    print(f"\n{str(datetime.now())} | Round {global_rounds + 1} Begin.")

    # Train the model on the subset with early stopping
    predator.fit(train_images, train_labels, epochs = predator_mini_epochs, verbose=1, batch_size=predator_batch_size, validation_data=(val_images,val_labels))


train_loss, train_acc = predator.evaluate(train_images, train_labels)

endtime = datetime.now()
stop_logging.set()
logging_thread.join()


print(f"{str(datetime.now())} | Train accuracy: {train_acc}")
print(f"That took {str(endtime - starttime)}")


import loggymclogface

timestamps, memory_usage, cpu_usage, gpu_usage, phases = loggymclogface.read_usage_log(logging_file_dir)
loggymclogface.plot_usage_graphs(timestamps, memory_usage, cpu_usage, gpu_usage, phases)


val_loss, val_acc = predator.evaluate(val_images, val_labels)

print(f"Validation accuracy: {val_acc}")

testdata_dict = unpickle(r'cifar-100-python/test')
test_images, test_labels = testdata_dict[b'data'], testdata_dict[b'fine_labels']
test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # resize

test_labels = tf.one_hot(test_labels,
                     depth=np.array(test_labels).max() + 1,
                     dtype=tf.float64)
test_labels = tf.squeeze(test_labels)

test_loss, test_acc = predator.evaluate(test_images, test_labels)

print(f"Test Accuracy: {test_acc}")
