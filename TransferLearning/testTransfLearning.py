import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import preprocess_input as pre_process_VGG
from tensorflow.python.keras.applications.resnet50 import preprocess_input as pre_process_ResNet
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as pre_process_Inception
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-m", "--model", required=True, help="Model used: 0 VGG, 1 ResNet, 2 Inception")
args = vars(ap.parse_args())
run_on_server = args["server"]
model_name = int(args["model"])

if run_on_server == "y":
        test_folder = ["/mnt/Data/ltanzi/flippedCrossVal/Testing/TestA",
                       "/mnt/Data/ltanzi/flippedCrossVal/Testing/TestB",
                       "/mnt/Data/ltanzi/flippedCrossVal/Testing/TestUnbroken"]
        score_folder = "/mnt/Data/ltanzi/flippedCrossVal/Test"
        model_path = "/mnt/Data/ltanzi/FlippedModels/Inception/"
elif run_on_server == "n":
        test_folder = ["/Users/leonardotanzi/Desktop/testEdgedA1A2A3/Testing/TestA1",
                       "/Users/leonardotanzi/Desktop/testEdgedA1A2A3/Testing/TestA2",
                       "/Users/leonardotanzi/Desktop/testEdgedA1A2A3/Testing/TestA3"]
        model_path = "/Users/leonardotanzi/Desktop/testEdgedA1A2A3/"
        score_folder = "/Users/leonardotanzi/Desktop/testEdgedA1A2A3/Test"
else:
        raise ValueError("Incorrect 1st arg.")


classmode = "sparse"
image_size = 299 if model_name == 2 else 224
batch_size = 1

if model_name == 0:
        preprocess_input = pre_process_VGG
elif model_name == 1:
        preprocess_input = pre_process_ResNet
elif model_name == 2:
        preprocess_input = pre_process_Inception


data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

classes = ["A", "B", "Unbroken"]
dict_classes = {classes[0]: 0, classes[1]: 1, classes[2]: 2}  

model = load_model(model_path + "Fold5_Flipped_lr00001-retrainAll-balanced-categorical-Inception-1569005609.model")

# Evaluate scores of the full test set

test_generator = data_generator.flow_from_directory(score_folder,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=classmode,
        classes=classes
)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


for i, folder in enumerate(test_folder):
        test_generator = data_generator.flow_from_directory(folder,
                                                            target_size=(image_size, image_size),
                                                            batch_size=batch_size,
                                                            class_mode=classmode)

        STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

        test_generator.reset()
        
        pred = model.predict_generator(test_generator,
                        steps=STEP_SIZE_TEST,
                        verbose=1)

        predicted_class_indices = np.argmax(pred, axis=1)

        labels = dict_classes
        labels = dict((v, k) for k, v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]

        print(predictions)

        x = 0
        tot = 0
        for j in predictions:
                tot += 1
                if j == classes[i]:
                        x += 1

        print("{} classified correctly: {}%".format(classes[i], x*100/tot))
