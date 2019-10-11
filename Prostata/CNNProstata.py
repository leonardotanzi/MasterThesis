import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import random
import time
from sklearn.model_selection import train_test_split
import argparse
import xlrd

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

image_size = 224

training_data = []

if run_on_server == "n":
    path = "/Users/leonardotanzi/Desktop/Dottorato/Prostata/"

else:
    path = "/mnt/data/ltanzi/Prostata/"


csv = "/mnt/data/ltanzi/MasterThesis/Prostata/data.xls"
train_path = path + "Train/"

wb = xlrd.open_workbook(csv) 
sheet = wb.sheet_by_index(0)   
sheet.cell_value(0, 0) 

for img_path in glob.glob(train_path + "*.png"):  # iterate over each image per broken and unbroken

    #img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #y = preprocess_input(x)

    for i in range(1, sheet.nrows):
        a = sheet.row_values(i)
        img_id = int(img_path.split("/")[-1].split(".")[0])
        img_id_excel = int(a[0])
        if img_id_excel == img_id:
            keypoints = [a[1], a[2]] #, a[3], a[4], a[5], a[6]]
            img_array = cv2.imread(img_path, cv2.COLOR_BGR2RGB)  # convert to array
            new_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
            training_data.append([new_array, keypoints])  # add this to our training_data
            break


random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

#qua ho due liste una con immagini e una con la label all'indice corrispondente (shufflate)

X = np.array(X).reshape(-1, image_size, image_size, 3)  # we need to convert x in numpy array, last 1 because it's grayscale

X = X/255.0  # normalize

X = preprocess_input(X)

model = Sequential()
model.add(ResNet50(include_top=False, input_shape=(image_size, image_size, 3), pooling="avg"))


#model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

'''
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''
'''
model.add(Dense(500))
model.add(Activation('relu'))
'''
model.add(Dense(2))

sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy"])
            
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)

model.save("prostata.model")

'''
model_path = path + "prostata.model"
model = load_model(model_path)
'''

x_test_path = path + "Test/576.png"
img = cv2.imread(x_test_path, cv2.COLOR_BGR2RGB)  # convert to array
x_test = cv2.resize(img, (image_size, image_size))  # resize to normalize data size

x_test = np.array(x_test).reshape(-1, image_size, image_size, 3)
x_test = x_test/255.0  # normalize
x_test = preprocess_input(x_test)

y_test = model.predict(x_test, batch_size=1)

print(y_test)
