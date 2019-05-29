import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]
run_binary = args["binary"]

if run_on_server == "y":
        test_folder = ["/mnt/Data/ltanzi/Train_Val/TestA", "/mnt/Data/ltanzi/Train_Val/TestB", "/mnt/Data/ltanzi/Train_Val/TestUnbroken"]
        model_path = "/home/ltanzi/"
elif run_on_server == "n":
        test_folder = ["/Users/leonardotanzi/Desktop/FinalDataset/TestA", "/Users/leonardotanzi/Desktop/FinalDataset/TestA", "/Users/leonardotanzi/Desktop/FinalDataset/TestA"]
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

dict_classes = {'Unbroken': 2, 'B': 1, 'A': 0}
classes = ["A", "B", "Unbroken"]

model = load_model(model_path + "transferLearning.model")

for i, folder in enumerate(test_folder):
        test_generator = data_generator.flow_from_directory(folder,
                                                            target_size=(image_size, image_size),
                                                            batch_size=24,
                                                            class_mode=classmode)

        STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


        score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        test_generator.reset()
        
        pred=model.predict_generator(test_generator,
                        steps=STEP_SIZE_TEST,
                        verbose=1)

        predicted_class_indices=np.argmax(pred,axis=1)

        labels = dict_classes
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]

        #  print(predictions)

        x = 0
        for j in predictions:
                if j == classes[i]:
                        x += 1

        print("{} classified correctly: {}%".format(classes[i], x))
