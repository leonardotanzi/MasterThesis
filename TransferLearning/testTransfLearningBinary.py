import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

class1 = "Broken"
class2 = "Unbroken"

if run_on_server == "y":
    score_folder = "/mnt/Data/ltanzi/Train_Val/Test"
    model_path = "/mnt/Data/ltanzi/"
    test_folder = ["/mnt/data/ltanzi/Train_Val/Testing/Test" + class1, "/mnt/data/ltanzi/Train_Val/Testing/Test" + class2]


elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/FinalDataset/"
    score_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Test"
    test_folder = ["/Users/leonardotanzi/Desktop/FinalDataset/Testing/Test" + class1, "/Users/leonardotanzi/Desktop/FinalDataset/Testing/Test" + class2]

else:
    raise ValueError("Incorrect arg.")


classmode = "binary"
image_size = 224
dict_classes = {class1: 0, class2: 1}
classes = [class1, class2]

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

model = load_model(model_path + "Broken_Unbroken-binary-baselineVGG-1562672986-best_model.h5")

# Evaluate scores of the full test set

test_generator = data_generator.flow_from_directory(score_folder,
                                                    target_size=(image_size, image_size),
                                                    batch_size=24,
                                                    class_mode=classmode,
                                                    classes=classes)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

# score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

for i, folder in enumerate(test_folder):

    print(folder)
    test_generator = data_generator.flow_from_directory(folder,
                                                        target_size=(image_size, image_size),
                                                        batch_size=24,
                                                        class_mode=classmode)

    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    test_generator.reset()

    pred = model.predict_generator(test_generator,
                                  steps=STEP_SIZE_TEST,
                                  verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

    sqArray = np.squeeze(pred)
    integer_predictions = []

    for p in sqArray:
        integer_predictions.append(int(round(p)))

    labels = dict_classes
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in integer_predictions]

    print(predictions)

    x = 0
    for j in predictions:
        if j == classes[i]:
            x += 1

    print("{} classified correctly: {}%".format(classes[i], x))
