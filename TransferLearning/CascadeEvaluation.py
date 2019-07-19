import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
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

class1 = "Broken"
class2 = "Unbroken"

subclass1 = "A"
subclass2 = "B"

if run_on_server == "y":
    score_folder = "/mnt/Data/ltanzi/Train_Val/Test"
    model_path = "/mnt/Data/ltanzi/"
    test_folder = ["/mnt/data/ltanzi/Train_Val/Testing/Test" + class1, "/mnt/data/ltanzi/Train_Val/Testing/Test" + class2]


elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/FinalDataset/"
    score_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val_CV/Test/Unbroken"
    test_folder = ["/Users/leonardotanzi/Desktop/FinalDataset/Testing/Test" + class1, "/Users/leonardotanzi/Desktop/FinalDataset/Testing/Test" + class2]

else:
    raise ValueError("Incorrect arg.")


output_path = "/Users/leonardotanzi/Desktop/Output/"

classmode = "binary"
image_size = 224
dict_classes = {class1: 0, class2: 1}
classes = [class1, class2]

data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, preprocessing_function=preprocess_input)

first_model = load_model(model_path + "Broken_Unbroken-binary-baselineVGG-1562672986.model")
second_model = load_model(model_path + "A_B-binary-baselineVGG-1562679455-best_model.h5")

i = 0

for img_path in sorted(glob.glob(score_folder + "/*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(224, 224))
    X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
           
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = first_model.predict(x)

    class_idx = int(round((preds[0][0])))
    if class_idx == 1:
        print("Unbroken")
        i += 1

    elif class_idx == 0:
        name_out = output_path + "{}".format(img_path.split("/")[-1])
        cv2.imwrite(name_out, X_original)


print("Unbroken {}".format(i))

i = 0
j = 0
for img_path in sorted(glob.glob(output_path + "*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(224, 224))
           
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = second_model.predict(x)

    class_idx = int(round((preds[0][0])))

    if class_idx == 0:
        print("A")
        i += 1

    elif class_idx == 1:
        print("B")
        j += 1


print("A {}".format(i))
print("B {}".format(j))