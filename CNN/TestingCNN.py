import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import random
import numpy as np


def print_img(name, img):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 370, 140)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


datadir= "/mnt/Data/ltanzi/Train_Val/Test"
categories = ["A", "B", "Unbroken"]
img_size = 256
training_data = []

for category in categories:

    path = os.path.join(datadir, category)  # create path to broken and unbroken
    class_num = categories.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1) # we need to convert x in numpy array, last 1 because it's grayscale

X = X/255.0

model = tf.keras.models.load_model("3-32-2.model")

score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
correct = 0
uncorrect = 0
unbroken_classified_as_broken = 0
broken_classified_as_unbroken = 0

prediction = model.predict([X])

for i in range(X.shape[0]):

    if int(round(prediction[i][0])) == int(y[i]):
        correct += 1
    else:
        uncorrect += 1

    label = CATEGORIES[int(y[i])]
    pred = CATEGORIES[int(round(prediction[i][0]))]
    name = "Label: {} / Prediction: {}".format(label, pred)
    print(name)

    if pred == 'Broken' and label == 'Unbroken':
        unbroken_classified_as_broken += 1

    if pred == 'Unbroken' and label == 'Broken':
        broken_classified_as_unbroken += 1

    # print_img(name, X[i])

tot = correct + uncorrect

print("Percentage: {}%, unbroken_classified_as_broken {} and broken_classified_as_unbroken {}".format(round(correct/tot*100, 4),
                                                                                                      unbroken_classified_as_broken,
                                                                                                      broken_classified_as_unbroken))
'''
