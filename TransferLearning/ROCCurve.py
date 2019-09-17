import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import random
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

def print_img(name, img):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 370, 140)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


if run_on_server == 'y':
    datadir = "/mnt/data/ltanzi/Train_Val_CV/Test"
    model_name = "/mnt/data/ltanzi/CV/Fold1_lr00001-batch32-notAugValTest-retrainAll-balanced-categorical-baselineInception-1563972372.model"
elif run_on_server == 'n':
    datadir = "/Users/leonardotanzi/Desktop/Test"
    model_name = "/Users/leonardotanzi/Desktop/Fold1_lr00001-batch32-notAugValTest-retrainAll-balanced-categorical-baselineInception-1563972372.model"

categories = ["A", "B", "Unbroken"]
img_size = 299
training_data = []
n_classes = 3
y_score = []

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


training_data = []

for category in categories:

    path = os.path.join(datadir, category)  # create path to broken and unbroken
    class_num = categories.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img = image.load_img(os.path.join(path, img), target_size=(img_size, img_size))
            training_data.append([img, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass


random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    x = image.img_to_array(features)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X.append(x)
    y.append(label)

y = label_binarize(y, classes=[0, 1, 2])


model = tf.keras.models.load_model(model_name)

for x in X:
    y_score.append(model.predict(x))
    
# y_score = model.predict(X)

'''
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

# X = np.array(X).reshape(-1, img_size, img_size, 1) # we need to convert x in numpy array, last 1 because it's grayscale

# X = X/255.0

for x in X:
    x = preprocess_input(x)

model = tf.keras.models.load_model("/Users/leonardotanzi/Desktop/Cascade/Fold4_IncV3-A_B-categorical-baselineInception-1568304568-best_model.h5")

score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")

if run_on_server == 'n':
    plt.show()
else:
    plt.savefig("/mnt/data/ltanzi/MasterThesis/TransferLearning/Roc1.png")


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")

if run_on_server == 'n':
    plt.show()
else:
    plt.savefig("/mnt/data/ltanzi/MasterThesis/TransferLearning/Roc2.png")
