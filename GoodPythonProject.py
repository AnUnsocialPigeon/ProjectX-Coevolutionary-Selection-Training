# Import and setup Weights and Biases
#import wandb

# Import required libraries
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# Load in the training data
file = os.getenv("CIFAR100_DIR")
train_data = unpickle(file)
print(type(train_data))
print(train_data.keys())

# meta_file = os.getenv("CIFAR10_DIR")
# meta_data = unpickle(meta_file)
# print(type(meta_data))
# print(meta_data.keys())

X_train = train_data['data']
# Reshape the whole image data
X_train = X_train.reshape(len(X_train),3,32,32)
# Transpose the whole data
X_train = X_train.transpose(0,2,3,1)


y_train = tf.one_hot(train_data['coarse_labels'],
                     depth=np.array(train_data['coarse_labels']).max() + 1,
                     dtype=tf.float64)
y_train = tf.squeeze(y_train)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
model = ResNet50(weights=None,input_shape=(32,32,3),classes=20)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

#wandb.init(project='projectX', entity='alistair-ryan')

hist = model.fit(X_train, y_train,
                 epochs=5,
                 batch_size=64,
                 verbose=1)

model_loss = pd.DataFrame(model.history.history)
model_loss.to_csv("model_loss.csv")
model.save("resnetcifar100.h5")