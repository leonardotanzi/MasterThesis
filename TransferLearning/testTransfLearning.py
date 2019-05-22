import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

if run_on_server == "y":
        datadir = "/mnt/Data/ltanzi/Train_Val/Test"
        model_path = "/mnt/Data/ltanzi/"
elif run_on_server == "n":
        datadir = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Test"
        model_path = "/Users/leonardotanzi/Desktop/FinalDataset/"
else:
        raise ValueError("Incorrect arg")


categories = ["A", "B", "Unbroken"]

image_size = 256

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(datadir,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')


model = load_model(model_path + "transferLearning.model")


score = model.evaluate(test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
