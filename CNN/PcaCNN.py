from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
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
        train_folder = "/mnt/Data/ltanzi/PcaCNN/Train"
        test_folder = "/mnt/Data/ltanzi/PcaCNN/Test"
        out_folder = "/mnt/Data/ltanzi/PcaCNN/"

elif run_on_server == "n":
        train_folder = "/Users/leonardotanzi/Desktop/NeededDataset/PcaCNN/Train"
        test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/PcaCNN/Test"
        out_folder = "/Users/leonardotanzi/Desktop/NeededDataset/PcaCNN/"

else:
        raise ValueError('Incorrect 1st arg.')


if run_binary == "y":
        last_layer = 2
        categories = ["Broken", "Unbroken"]
        loss = "sparse_categorical_crossentropy"

elif run_binary == "n":
        last_layer = 3
        categories = ["A", "B", "Unbroken"]
        loss = "sparse_categorical_crossentropy"

else:
        raise ValueError('Incorrect 2nd arg.')


image_size = 256
cannyWindow = 17

training_data = []

for category in categories:

    path = os.path.join(train_folder, category)  # create path to broken and unbroken
    class_num = categories.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            resized_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
            edged_array = cv2.cv2.Canny(resized_array, cannyWindow, cannyWindow * 3, apertureSize=3)
            training_data.append([edged_array, class_num])  # add this to our training_data
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

'''
numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). It simply means that it 
is an unknown dimension and we want numpy to figure it out. And numpy will figure this by looking at the 'length of the 
array and remaining dimensions' and making sure it satisfies the above mentioned criteria.
z = np.array([[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]])
        
New shape as (-1, 2). row unknown, column 2. we get result new shape as (6, 2)
z.reshape(-1, 2)
array([[ 1,  2],
   [ 3,  4],
   [ 5,  6],
   [ 7,  8],
   [ 9, 10],
   [11, 12]])

It means, that the size of the dimension, for which you passed -1, is being inferred. Thus,

A.reshape(-1, 28*28)
means, "reshape A so that its second dimension has a size of 28*28 and calculate the correct size of the first dimension".

'''

X = X/255.0  # normalize

model_name = "BaselineNet-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

model = Sequential()

model.add(Dense(32, input_shape=X.shape[1:])) # X.shape[] prende la seconda a la terza shape che sono altezza e larghezza immagine
model.add(Activation("relu"))

model.add(Dense(last_layer, activation="softmax"))

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)  # the optimizer, as the sgd

model.compile(loss=loss,
              optimizer=adam,
              metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.33, callbacks=[tensorboard])

model.summary()

model.save("{}.model".format(model_name))


# Test

for category in categories:

    path = os.path.join(test_folder, category)  # create path to broken and unbroken
    class_num = categories.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            resized_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
            edged_array = cv2.cv2.Canny(resized_array, cannyWindow, cannyWindow * 3, apertureSize=3)
            training_data.append([edged_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass

pass
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, image_size, image_size, 1)

X = X / 255.0

score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
