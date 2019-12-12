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
from tensorflow.python.keras.applications.vgg16 import preprocess_input as pre_process_VGG
from tensorflow.python.keras.applications.resnet50 import preprocess_input as pre_process_ResNet
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as pre_process_Inception
from sklearn.preprocessing import label_binarize
from keras.preprocessing import image
import glob
import argparse
import scipy


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


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
    ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
    ap.add_argument("-m", "--model", required=True, help="Select the network (0 for VGG, 1 for ResNet, 2 for InceptionV3)")
    args = vars(ap.parse_args())
    run_on_server = args["server"]
    run_binary = args["binary"]
    run_model = int(args["model"])

    models = ["VGG", "ResNet", "Inception"]
    model_type = models[run_model]
    img_size = 224 if run_model == 0 or run_model == 1 else 299
    
    if model_type == "VGG":
        preprocess_input = pre_process_VGG
    elif model_type == "ResNet":
        preprocess_input = pre_process_ResNet
    elif model_type == "Inception":
        preprocess_input = pre_process_Inception
        
    if run_on_server == 'y':
        datadir = "/mnt/data/ltanzi/PAPER/All_Cross_Val/Test"
        model_path = "/mnt/data/ltanzi/PAPER/Output/Cascade/Models/" # "/mnt/data/ltanzi/PAPER/Output/Classic/{}/5classes/Models/".format(model_type)
        out_path = "/mnt/data/ltanzi/PAPER/Output/Cascade/MetricSingleNet/AB/"# "/mnt/data/ltanzi/PAPER/Output/Classic/{}/5classes/Metrics/Normal/".format(model_type)

    elif run_on_server == 'n':
        datadir = "/Users/leonardotanzi/Desktop/Test"
        model_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/"
        out_path = "/Users/leonardotanzi/Desktop/"

    if run_binary == "n":
        classes = ["A1", "A2", "A3", "B", "Unbroken"]
    elif run_binary =="y":
        classes = ["A", "B"]
        
    training_data = []
    n_classes = len(classes)
    n_fold = 5
    accuracies = []
    precisions = [[] for x in range(n_classes)]
    recalls = [[] for x in range(n_classes)]
    f1scores = [[] for x in range(n_classes)]
    y_score_ROC = []
    roc_avg = [[] for x in range(n_classes)]

    for category in classes:

        path = os.path.join(datadir, category)  # create path to broken and unbroken
        class_num = classes.index(category)
        for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
            if img.endswith(".png"):
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

    if run_binary == "n":
        y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    elif run_binary == "y":
        y = label_binarize(y, classes=[0, 1])

    y_ROC = np.concatenate((y, y, y, y, y), axis=0)

    # model_name = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Fold3_A1A2A3_notflipped-retrainAll-categorical-Inception-1569509422.model"
    # model = tf.keras.models.load_model(model_name)

    for fold_n in range(n_fold):

        model_name = model_path + "Fold{}_Inception_AB-best_model.h5".format(fold_n + 1)
        model = tf.keras.models.load_model(model_name)
        y_score = []

        print("\n\nFold number {}".format(fold_n + 1))
        
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

        for i in range(n_classes):
            precisions[i].append(classification_report_dict['{}'.format(i)]['precision'])
            recalls[i].append(classification_report_dict['{}'.format(i)]['recall'])
            f1scores[i].append(classification_report_dict['{}'.format(i)]['f1-score'])

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
            roc_avg[i].append(roc_auc[i])

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

        plt.savefig(out_path + "Fold{}_ROC.png".format(fold_n + 1))
        plt.close()

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

        plt.savefig(out_path + "Fold{}_ROCzoom.png".format(fold_n + 1))
        plt.close()
        '''

    print("\n\nAveraged results among 5 folds:")
    # Print accuracies
    mean_acc, CI_acc_low, CI_acc_high = mean_confidence_interval(accuracies)
    print("Avg accuracy: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(mean_acc, CI_acc_low, CI_acc_high))

    # Print precision, recall, f1-score
    for i in range(n_classes):
        mean_prec, CI_prec_low, CI_prec_high = mean_confidence_interval(precisions[i])
        mean_rec, CI_rec_low, CI_rec_high = mean_confidence_interval(recalls[i])
        mean_f1, CI_f1_low, CI_f1_high = mean_confidence_interval(f1scores[i])

        print("Avg precision class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_prec, CI_prec_low, CI_prec_high))
        print("Avg recall class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_rec, CI_rec_low, CI_rec_high))
        print("Avg f1-score class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_f1, CI_f1_low, CI_f1_high))

    for i in range(n_classes):
        mean_auc, CI_low, CI_high = mean_confidence_interval(roc_avg[i])
        print("Avg AUC class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_auc, CI_low, CI_high))
    '''
    # print avg roc
    lw = 2
    y_score_ROC = np.squeeze(y_score_ROC)
    # Compute ROC curve and ROC area for each class
    fpr_avg = dict()
    tpr_avg = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr_avg[i], tpr_avg[i], _ = roc_curve(y_ROC[:, i], y_score_ROC[:, i])
        roc_auc[i] = auc(fpr_avg[i], tpr_avg[i])

    # Compute micro-average ROC curve and ROC area
    fpr_avg["micro"], tpr_avg["micro"], _ = roc_curve(y_ROC.ravel(), y_score_ROC.ravel())
    roc_auc["micro"] = auc(fpr_avg["micro"], tpr_avg["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr_avg = np.unique(np.concatenate([fpr_avg[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr_avg)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr_avg, fpr_avg[i], tpr_avg[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr_avg["macro"] = all_fpr_avg
    tpr_avg["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr_avg["macro"], tpr_avg["macro"])

    # Plot all ROC curves
    plt.figure(1)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr_avg[i], tpr_avg[i], color=color, lw=lw,
                 label='ROC of {0} class (AUC = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Averaged ROC')
    plt.legend(loc="lower right")

    plt.savefig(out_path + "AvgROC.png")
    '''
