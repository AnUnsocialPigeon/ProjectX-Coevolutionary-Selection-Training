import random
import time
from deap import base, creator, tools

# Predator libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


def train_predators(model, given, expected_output, epochs=4):
    model.fit(given, expected_output, epochs=epochs, verbose=2)