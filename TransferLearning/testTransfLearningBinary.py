import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

if run_on_server == "y":
    score_folder = "/mnt/Data/ltanzi/A_B/Test"
    model_path = "/mnt/Data/ltanzi/"

elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/FinalDataset/"
    score_folder = "/Users/leonardotanzi/Desktop/FinalDataset/BinaryDatasets/Bro_Unbro/Test"
else:
    raise ValueError("Incorrect arg.")


classmode = "binary"
image_size = 256
class1 = "A"
class2 = "B"

data_generator = ImageDataGenerator(zca_whitening=True, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,preprocessing_function=preprocess_input)

dict_classes = {class2: 1, class1: 0}
classes = [class1, class2]

model = load_model(model_path + "transferLearningAB.model")

# Evaluate scores of the full test set

test_generator = data_generator.flow_from_directory(score_folder,
                                                    target_size=(image_size, image_size),
                                                    batch_size=24,
                                                    class_mode=classmode)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

