from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Activation
# from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import backend as K
import argparse
import numpy as np
import time
import os
import tensorflow as tf


if __name__ == "__main__":

    image_size = 224
    binary = "categorical"
    loss = "sparse_categorical_crossentropy"
    last_layer = 3
    classmode = "sparse"
    act = "softmax"
    classes = ["XR_FOREARM", "XR_HAND", "XR_HUMERUS"]

    train_folder = "/mnt/data/ltanzi/MURA/train"
    val_folder = "/mnt/data/ltanzi/MURA/valid"
    out_folder = "/mnt/data/ltanzi/MURA/"
    name = "pre_trained_weights_MURA"

    tensorboard = TensorBoard(log_dir="/mnt/data/ltanzi/CV/logs/{}".format(name))
    es = EarlyStopping(monitor="val_acc", mode="max", verbose=1,
                       patience=10)  # verbose to print the n of epoch in which stopped,
    best_model_path = out_folder + name + "-best_model.h5"
    mc = ModelCheckpoint(best_model_path, monitor="val_acc", save_best_only=True, mode='max', verbose=1)

    my_new_model = Sequential()
    my_new_model.add(VGG16(include_top=False, input_shape=(image_size, image_size, 3), pooling="avg", weights=None))
    my_new_model.add(Dense(last_layer, activation=act))
    my_new_model.layers[0].trainable = True

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)

    my_new_model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])
    data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
            horizontal_flip=True, preprocessing_function=preprocess_input)
    data_generator_notAug = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = data_generator.flow_from_directory(train_folder,
            target_size=(image_size, image_size),
            batch_size=32,
            class_mode=classmode,
            classes=classes)

    validation_generator = data_generator_notAug.flow_from_directory(val_folder,
            target_size=(image_size, image_size),
            batch_size=32,
            class_mode=classmode,
            classes=classes)
  
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
    my_new_model.fit_generator(
                        train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=150,
                        validation_data=validation_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[tensorboard, es, mc])

    my_new_model.save(out_folder + name + ".model")
