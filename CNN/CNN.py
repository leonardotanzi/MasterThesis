import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import time
from sklearn.model_selection import train_test_split
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]
run_binary = args["binary"]

if run_on_server == "y" and run_binary == "y":
        train_folder = "/mnt/Data/ltanzi/Train_Val_BROUNBRO/Train"
        val_folder = "/mnt/Data/ltanzi/Train_Val_BROUNBRO/Validation"
        out_folder = "/mnt/Data/ltanzi/"
        resnet_weights_path = "imagenet"
        categories = ["B", "Unbroken"]
        num_classes = 2

elif run_on_server == "n":
        train_folder = "/Users/leonardotanzi/Desktop/FinalDataset2/Train_Val/Train"
        val_folder = "/Users/leonardotanzi/Desktop/FinalDataset2/Train_Val/Validation"
        out_folder = "/Users/leonardotanzi/Desktop/FinalDataset/"
        resnet_weights_path = "/Users/leonardotanzi/Desktop/MasterThesis/TransferLearning/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

else:
        raise ValueError('Incorrect arg')


categories = ["A", "B", "Unbroken"]

image_size = 256

num_classes = 3

training_data = []

for category in categories:

    path = os.path.join(train_folder, category)  # create path to broken and unbroken
    class_num = categories.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass

pass
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, image_size, image_size, 1)  # we need to convert x in numpy array, last 1 because it's grayscale

X = X/255.0

for img in X:
    a = 1

conv_layers = [2]  # 3
layer_sizes = [32]  # 64
dense_layers = [2]  # 0


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            NAME = "{}conv-{}nodes-{}dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            print(NAME)
            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:])) #64 is the number of filter used
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):  # because we need for sure 1 conv layer
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                # model.add(Dropout(0.25))

            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
                # model.add(Dropout(0.5))

            model.add(Dense(num_classes))
            model.add(Activation("sigmoid"))

            adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)

            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer="sgd",
                          metrics=["accuracy"])

            model.fit(X, y, batch_size=32, epochs=50, validation_split=0.3, callbacks=[tensorboard])

            model.summary()
            # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

            model.save("2-32-2new.model")
