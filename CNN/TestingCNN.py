import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import random
'''
def prepare(filepath):
    IMG_SIZE = 256
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
'''

def print_img(name, img):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 370, 140)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


DATADIR = "/Users/leonardotanzi/Desktop/MasterThesis/CNN/Test"

CATEGORIES = ["Broken", "Unbroken"]

IMG_SIZE = 256

training_data = []


for category in CATEGORIES:

    path = os.path.join(DATADIR, category)  # create path to broken and unbroken
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass


random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


model = tf.keras.models.load_model("firstModel.model")

correct = 0
uncorrect = 0

for img, label in zip(X, y):

    # always made prediction on list that's why we have []
    prediction = model.predict([img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])
    if prediction == label:
        correct += 1
    else:
        uncorrect += 1

    name = "Label: {} / Prediction: {}".format(CATEGORIES[int(label)], CATEGORIES[int(prediction[0][0])])

    print(prediction[0][0])
    # print_img(name, img)

tot = correct + uncorrect

print("Percentage: {}%".format(round(correct/tot*100, 4)))

