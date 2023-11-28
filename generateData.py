import random
import time
from deap import base, creator, tools

# Predator libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def generate_data(num_samples, array_size):
    given = np.random.choice(['A', 'B', 'C'], size=(num_samples, array_size))
    expected = (given == 'B').astype(int) + (2 * (given == 'A').astype(int))
    return given, expected