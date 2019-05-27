import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]
run_binary = args["binary"]

if run_on_server == "y":
        datadir = "/mnt/Data/ltanzi/Train_Val/TestUnbroken"
        model_path = "/mnt/Data/ltanzi/"
elif run_on_server == "n":
        datadir = "/Users/leonardotanzi/Desktop/FinalDataset/TestA"
        model_path = "/Users/leonardotanzi/Desktop/FinalDataset/"
else:
        raise ValueError("Incorrect 1st arg.")

if run_binary == "y":
        classmode = 'binary'
elif run_binary == "n":
        classmode = "sparse"
else:
        raise ValueError("Incorrect 2nd arg.")

image_size = 256

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(datadir,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode=classmode)

pass
model = load_model(model_path + "transferLearning.model")


score = model.evaluate_generator(test_generator, steps=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict_generator(test_generator, steps=1)
indexes = tf.argmax(predictions, axis=1)
for i in range(indexes.shape[0]):
        print(indexes[i])
