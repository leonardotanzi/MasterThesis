import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import time

DATADIR = "/Users/leonardotanzi/Desktop/MasterThesis/CNN"

CATEGORIES = ["Broken", "Unbroken"]

# NAME = "FirstNN-{}".format(int(time.time()))
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  # create path to broken and unbroken
    for img in os.listdir(path):  # iterate over each image per broken and unbroken
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array

        break
    break

IMG_SIZE = 256
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')
# plt.show()

training_data = []

for category in CATEGORIES:

    path = os.path.join(DATADIR, category)  # create path to broken and unbroken
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass
        # except OSError as e:
        #    print("OSErrroBad img most likely", e, os.path.join(path,img))
        # except Exception as e:
        #    print("general exception", e, os.path.join(path,img))


random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # we need to convert x in numpy array, last 1 because it's grayscale

X = X/255.0

conv_layers = [3]
layer_sizes = [64]
dense_layers = [0]


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

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",
                          optimizer="adam",
                          metrics=["accuracy"])

            model.fit(X, y, batch_size=32, epochs=15, validation_split=0.3, callbacks=[tensorboard])

            model.summary()
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

            model.save("firstModel.model")