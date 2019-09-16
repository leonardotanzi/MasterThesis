# can evaluate one single folder a time

import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import argparse
import numpy as np
import glob
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

# class1 = "Broken"
# class2 = "Unbroken"

subclass1 = "A"
subclass2 = "B"
subclass3 = "Unbroken"

if run_on_server == "y":
    score_folder = "/mnt/Data/ltanzi/Train_Val/Test"
    model_path = "/mnt/Data/ltanzi/"
    test_folder = ["/mnt/data/ltanzi/Train_Val/Testing/Test" + class1, "/mnt/data/ltanzi/Train_Val/Testing/Test" + class2]


elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/Cascade/"
    score_folder = "/Users/leonardotanzi/Desktop/Cascade/Test/Unbroken"
    score_folder_A1A2A3 = "/Users/leonardotanzi/Desktop/SubgroupA_Proportioned/Test/A3"
    test_folder = ["/Users/leonardotanzi/Desktop/Cascade/Testing/Test" + subclass1,
                   "/Users/leonardotanzi/Desktop/Cascade/Testing/Test" + subclass2,
                   "/Users/leonardotanzi/Desktop/Cascade/Testing/Test" + subclass3]
else:
    raise ValueError("Incorrect arg.")


output_path = "/Users/leonardotanzi/Desktop/Cascade/OutputBroUnbro/"
output_path_AB = "/Users/leonardotanzi/Desktop/Cascade/OutputAB/"


classmode = "sparse"
image_size = 299
# dict_classes = {class1: 0, class2: 1}
# classes = [class1, class2]

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

first_model = load_model(model_path + "Fold1_IncV3-Broken_Unbroken-categorical-baselineInception-1568367921-best_model.h5")
second_model = load_model(model_path + "Fold4_IncV3-A_B-categorical-baselineInception-1568304568-best_model.h5")
third_model = load_model(model_path + "Fold1_A1A2A3balanced-retrainAll-categorical-Inception-1568302055-best_model.h5")

i = 0
j = 0

for img_path in sorted(glob.glob(score_folder_A1A2A3 + "/*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(image_size, image_size))
    X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
           
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = first_model.predict(x)

    class_idx = np.argmax(preds, axis=1)

    if class_idx == 1:
        print("Unbroken")
        i += 1

    elif class_idx == 0:
        print("Broken")
        j += 1
        name_out = output_path + "{}".format(img_path.split("/")[-1])
        cv2.imwrite(name_out, X_original)


print("Unbroken {} - Broken {}".format(i, j))

i = 0
j = 0
for img_path in sorted(glob.glob(output_path + "*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(image_size, image_size))
    X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = second_model.predict(x)

    class_idx = np.argmax(preds, axis=1)

    if class_idx == 0:
        print("A")
        i += 1
        name_out = output_path_AB + "{}".format(img_path.split("/")[-1])
        cv2.imwrite(name_out, X_original)

    elif class_idx == 1:
        print("B")
        j += 1


print("A {} - B {}".format(i, j))


i = 0
j = 0
k = 0

for img_path in sorted(glob.glob(output_path_AB + "*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(image_size, image_size))
    X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = third_model.predict(x)

    class_idx = np.argmax(preds, axis=1)

    if class_idx == 0:
        print("A1")
        i += 1

    elif class_idx == 1:
        print("A2")
        j += 1

    elif class_idx == 2:
        print("A3")
        k += 1

print("A1 {} - A2 {} - A3 {}".format(i, j, k))