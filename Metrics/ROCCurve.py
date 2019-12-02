import tensorflow as tf
import cv2
import os
from tqdm import tqdm
import random
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import label_binarize
import glob
import argparse
import scipy

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


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


if run_on_server == 'y':
    datadir = "/mnt/data/ltanzi/Train_Val_CV/Test"
    model_path = "/mnt/data/ltanzi/"
    out_path = "/mnt/data/ltanzi/MasterThesis/Metrics/"

elif run_on_server == 'n':
    datadir = "/Users/leonardotanzi/Desktop/Test"
    model_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/"
    out_path = "/Users/leonardotanzi/Desktop/"

classes = ["A1", "A2", "A3"]
img_size = 299
training_data = []
n_classes = 3
accuracies = []
precisions = [[] for x in range(n_classes)]
recalls = [[] for x in range(n_classes)]
f1scores = [[] for x in range(n_classes)]
y_score_ROC = []

for category in classes:

    path = os.path.join(datadir, category)  # create path to broken and unbroken
    class_num = classes.index(category)  # get the classification  (0 or a 1). 0=broken 1=unbroken

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

model_name = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Fold3_A1A2A3_notflipped-retrainAll-categorical-Inception-1569509422.model"
model = tf.keras.models.load_model(model_name)

for fold_n in range(5):

    #model_name = model_path + "Fold{}_modelblabla.model".format(fold_n)
    y_score = []

    print("Model {}".format(model_name))
    # model = tf.keras.models.load_model(model_name)

    for x in X:
        pred = model.predict(x)
        y_score.append(pred)
        y_score_ROC.append(pred)

    y_score = np.squeeze(y_score)
    actual_values = y.argmax(axis=1)
    predicted_values = y_score.argmax(axis=1)

    matrix = confusion_matrix(actual_values, predicted_values)
    accuracy = accuracy_score(actual_values, predicted_values)
    classification_report_out = classification_report(y_true=actual_values, y_pred=predicted_values)
    classification_report_dict = classification_report(actual_values, predicted_values, output_dict=True)

    print('Confusion Matrix :')
    print(matrix)
    print('Accuracy Score:', )
    print(accuracy)
    print('Classification Report : ')
    print(classification_report_out)

    accuracies.append(accuracy)

    for i in range(n_classes)
        precisions[i].append(classification_report_dict['{}'.format(i)]['precision'])
        recalls[i].append(classification_report_dict['{}'.format(i)]['recall')]
        f1scores[i].append(classification_report_dict['{}'.format(i)]['f1-score'])
    recall1 = classification_report_dict['0']['recall']
    recalls[0].append(recall1)
    recall2 = classification_report_dict['1']['recall']
    recalls[1].append(recall2)
    recall3 = classification_report_dict['2']['recall']
    recalls[2].append(recall2)

    f1score1 = classification_report_dict['0']['f1-score']
    f1scores[0].append(f1score1)
    f1score2 = classification_report_dict['1']['f1-score']
    f1scores[1].append(f1score2)
    f1score3 = classification_report_dict['2']['f1-score']
    f1scores[2].append(f1score2)


    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    roc_avg = [for i in classes]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        roc_tot += roc_auc[i]

    roc_mean =
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
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC of {0} class (AUC = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Fold{}'.format(fold_n))
    plt.legend(loc="lower right")

    plt.savefig(out_path + "Fold{}_ROC.png".format(fold_n))

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC of {0} class (AUC = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Fold{}'.format(fold_n))
    plt.legend(loc="lower right")

    plt.savefig(out_path + "Fold{}zoom.png".format(fold_n))

mean_acc, CI_acc_low, CI_acc_high = mean_confidence_interval(accuracies)

mean_prec_1, CI_prec_low_1, CI_prec_high_1 = mean_confidence_interval(precisions[0])
mean_prec_2, CI_prec_low_2, CI_prec_high_2 = mean_confidence_interval(precisions[1])
mean_prec_3, CI_prec_low_3, CI_prec_high_3 = mean_confidence_interval(precisions[2])

mean_rec_1, CI_rec_low_1, CI_rec_high_1 = mean_confidence_interval(recalls[0])
mean_rec_2, CI_rec_low_2, CI_rec_high_2 = mean_confidence_interval(recalls[1])
mean_rec_3, CI_rec_low_3, CI_rec_high_3 = mean_confidence_interval(recalls[2])

mean_f1_1, CI_f1_low_1, CI_f1_high_1 = mean_confidence_interval(f1scores[0])
mean_f1_2, CI_f1_low_2, CI_f1_high_2 = mean_confidence_interval(f1scores[1])
mean_f1_3, CI_f1_low_3, CI_f1_high_3 = mean_confidence_interval(f1scores[2])

print("Avg accuracy: {} (CI {}-{}\n".format(mean_acc, CI_acc_low, CI_acc_high))

print("Avg precision class {}: {} (CI {}-{}\n".format(classes[0], mean_prec_1, CI_prec_low_1, CI_prec_high_1))
print("Avg precision class {}: {} (CI {}-{}\n".format(classes[1], mean_prec_2, CI_prec_low_2, CI_prec_high_2))
print("Avg precision class {}: {} (CI {}-{}\n".format(classes[2], mean_prec_3, CI_prec_low_3, CI_prec_high_3))

print("Avg recall class {}: {} (CI {}-{}\n".format(classes[0], mean_rec_1, CI_rec_low_1, CI_rec_high_1))
print("Avg recall class {}: {} (CI {}-{}\n".format(classes[1], mean_rec_2, CI_rec_low_2, CI_rec_high_2))
print("Avg recall class {}: {} (CI {}-{}\n".format(classes[2], mean_rec_3, CI_rec_low_3, CI_rec_high_3))

print("Avg f1-score class {}: {} (CI {}-{}\n".format(classes[0], mean_f1_1, CI_f1_low_1, CI_f1_high_1))
print("Avg f1-score class {}: {} (CI {}-{}\n".format(classes[1], mean_f1_2, CI_f1_low_2, CI_f1_high_2))
print("Avg f1-score class {}: {} (CI {}-{}\n".format(classes[2], mean_f1_3, CI_f1_low_3, CI_f1_high_3))