import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator




'''

NUM_CLASSES = 2
CHANNELS = 1
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
# Common accuracy metric for all outputs, but can use different metrics for different output
LOSS_METRICS = ['accuracy']
# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10
# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively
# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 32
# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()

model.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights=resnet_weights_path))

model.add(Dense(NUM_CLASSES, activation=DENSE_LAYER_ACTIVATION))

model.layers[0].trainable = False

model.summary()

sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

image_size = IMAGE_RESIZE
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = data_generator.flow_from_directory('Broken',
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE_TRAINING,
        class_mode='categorical')

'''