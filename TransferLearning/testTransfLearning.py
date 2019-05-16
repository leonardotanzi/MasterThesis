import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


DATADIR = "/Users/leonardotanzi/Desktop/FinalDataset/Train_val/Test"

CATEGORIES = ["A", "B", "Unbroken"]

image_size = 256


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(DATADIR,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')


model = tf.keras.models.load_model("transferLearning.model")


score = model.evaluate(test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])