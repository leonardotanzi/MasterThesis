from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.optimizers import Adam

import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]
run_binary = args["binary"]

if run_on_server == "y":
        train_folder = "/mnt/Data/ltanzi/A1A2A3onefold/Train"
        test_folder = "/mnt/Data/ltanzi/A1A2A3onefold/Test"
        out_folder = "/mnt/Data/ltanzi/A1A2A3/pca/"

elif run_on_server == "n":
        train_folder = "/Users/leonardotanzi/Desktop/NeededDataset/PcaCNNsmall/Train"
        test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/PcaCNNsmall/Test"
        out_folder = "/Users/leonardotanzi/Desktop/NeededDataset/PcaCNNsmall/"

else:
        raise ValueError('Incorrect 1st arg.')


if run_binary == "y":
        last_layer = 2
        categories = ["Broken", "Unbroken"]
        loss = "sparse_categorical_crossentropy"
        n_class = 2

elif run_binary == "n":
        last_layer = 3
        categories = ["A1", "A2", "A3"]
        loss = "sparse_categorical_crossentropy"
        n_class = 3

else:
        raise ValueError('Incorrect 2nd arg.')


image_size = 256
cannyWindow = 17

training_data = []
testing_data = []

for category in categories:

    path = os.path.join(train_folder, category)  # create path to broken and unbroken
    class_num = categories.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            resized_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
            # edged_array = cv2.Canny(resized_array, cannyWindow, cannyWindow * 3, apertureSize=3)
            training_data.append([resized_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass

pass
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#qua ho due liste una con immagini e una con la label all'indice corrispondente (shufflate)

X = np.array(X).reshape(-1, image_size, image_size, 1)  # we need to convert x in numpy array, last 1 because it's grayscale

model_name = "EdgedNet-lr00001{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))
es = EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=30)  # verbose to print the n of epoch in which stopped,


dimData = np.prod(X.shape[1:])
X = X.reshape(X.shape[0], dimData)
X = X.astype('float32')
X = X/255.0  # normalize

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(n_class, activation='softmax'))

model.summary()

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)  # the optimizer, as the sgd

model.compile(loss=loss,
              optimizer=adam,
              metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=100, validation_split=0.33, callbacks=[tensorboard, es])

model.save("{}.model".format(model_name))


# Test

for category in categories:

    path = os.path.join(test_folder, category)  # create path to broken and unbroken
    class_num = categories.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            resized_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
            # edged_array = cv2.Canny(resized_array, cannyWindow, cannyWindow * 3, apertureSize=3)
            testing_data.append([resized_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass

pass
random.shuffle(testing_data)

X = []
y = []

for features, label in testing_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, image_size, image_size, 1)

dimData = np.prod(X.shape[1:])
X = X.reshape(X.shape[0], dimData)
X = X.astype('float32')
X = X/255.0  # normalize

score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
