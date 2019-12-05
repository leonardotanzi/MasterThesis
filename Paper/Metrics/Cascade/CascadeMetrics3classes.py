import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
import argparse
import numpy as np
import glob
import cv2
import os
import shutil
import scipy
from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


if __name__ == "__main__":


    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
    args = vars(ap.parse_args())
    run_on_server = args["server"]

    classes = ["A", "B", "Unbroken"]
    n_classes = len(classes)
    n_folds = 5
    image_size = 299
    ground_truth_dict = {}
    final_predictions_dict = {}
    y_score_dict = {}

    # in order to avg the values for each class and fold
    accuracies = []
    precisions = [[] for x in range(n_classes)]
    recalls = [[] for x in range(n_classes)]
    f1scores = [[] for x in range(n_classes)]
    y_score_ROC = []
    roc_avg = [[] for x in range(n_classes)]

    if run_on_server == "y":
        model_path = "/mnt/Data/ltanzi/Cascade/BestModels/"
        test_folder = "/mnt/data/ltanzi/Train_Val_CV/Test/"
        output_path = "/mnt/data/ltanzi/Cascade/OutputBroUnbro/"
        output_path_AB = "/mnt/data/ltanzi/Cascade/OutputAB/"
        file_path = "/mnt/data/ltanzi/Cascade/metricsBest.txt"
        out_path = "/mnt/data/ltanzi/Cascade/ROC/"

    elif run_on_server == "n":
        model_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/"
        test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Test/"
        output_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/OutputBroUnbro/"
        output_path_AB = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/OutputAB/"
        file_path = "/Users/leonardotanzi/Desktop/metrics.txt"
        out_path = "/Users/leonardotanzi/Desktop/"

        # score_folder_A1A2A3 = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Proportioned/Test/A3"
        # test_folder_A1A2A3 = ["/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass1,
        #               "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass2,
        #               "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass3]
    else:
        raise ValueError("Incorrect arg.")

    for class_n in range(n_classes):
        for img_path in sorted(glob.glob(test_folder + "{}/*.png".format(classes[class_n])), key=os.path.getsize):
            original_name = img_path.split("/")[-1].split(".")[0]
            ground_truth_dict["{}".format(original_name)] = class_n
            final_predictions_dict["{}".format(original_name)] = -1
            # y_score_dict["{}".format(original_name)] = (0, 0, 0)  se mi servisse fare AUC anche per cascata, dovrei
            # far passare tutte le imm per bro-unbro, salvare le score per unbro, e poi farle passare tutte
            # (non solo le bro) per A-B e risalvarmi le score.

    for fold_n in range(n_folds):
        # first_model = load_model(model_path + "Fold{}_BroUnbro.h5".format(fold_n + 1))
        # second_model = load_model(model_path + "Fold{}_AB.h5".format(fold_n + 1))
        # y_score = []

        first_model = load_model("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Fold1_IncV3-Broken_Unbroken-categorical-baselineInception-1568367921-best_model.h5")
        second_model = load_model("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Fold4_IncV3-A_B-categorical-baselineInception-1568304568-best_model.h5")

        # conf_matrix = np.zeros((n_classes, n_classes))

        for class_n in range(n_classes):
            for img_path in sorted(glob.glob(test_folder + "{}/*.png".format(classes[class_n])), key=os.path.getsize):

                img = image.load_img(img_path, target_size=(image_size, image_size))
                X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
                original_name = img_path.split("/")[-1].split(".")[0]

                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = first_model.predict(x)
                # y_score_ROC.append(preds)

                class_idx = np.argmax(preds, axis=1)

                if class_idx == 1:
                    print("Unbroken")
                    # conf_matrix[class_n][2] += 1
                    final_predictions_dict["{}".format(original_name)] = 2
                    # y_score_dict["{}".format(original_name)] = preds

                elif class_idx == 0:
                    print("Broken")
                    name_out = output_path + "{}".format(img_path.split("/")[-1])
                    cv2.imwrite(name_out, X_original)

            for img_path in sorted(glob.glob(output_path + "*.png"), key=os.path.getsize):

                img = image.load_img(img_path, target_size=(image_size, image_size))
                X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
                original_name = img_path.split("/")[-1].split(".")[0]

                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = second_model.predict(x)

                class_idx = np.argmax(preds, axis=1)

                if class_idx == 0:
                    print("A")
                    # conf_matrix[class_n][0] += 1
                    final_predictions_dict["{}".format(original_name)] = 0
                    # cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/APredicted/{}-Label{}-PredictedA.png".format(original_name, label), X_original)
                    # name_out = output_path_AB + "{}".format(img_path.split("/")[-1])
                    # cv2.imwrite(name_out, X_original)

                elif class_idx == 1:
                    print("B")
                    # conf_matrix[class_n][1] += 1
                    final_predictions_dict["{}".format(original_name)] = 1
                    # cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/BPredicted/{}-Label{}-PredictedB.png".format(original_name, label), X_original)

                # y_score_dict["{}".format(original_name)] = preds

            shutil.rmtree(output_path)
            shutil.rmtree(output_path_AB)
            os.mkdir(output_path)
            os.mkdir(output_path_AB)

        #PARTIRE DA QUA, VEDERE COME CONVERTIRE I DICT IN NDARRAY
        '''
        y_score = []
        for i in y_score_dict:
            y_score.append(y_score_dict[i])

        y_score = np.asarray(y_score)
        '''
        actual_values = []
        for i in ground_truth_dict:
            actual_values.append(ground_truth_dict[i])

        predicted_values = []
        for i in final_predictions_dict:
            predicted_values.append(final_predictions_dict[i])

        #print(conf_matrix)
        '''
        actual_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        predicted_values = [0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        '''
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
        y = label_binarize(actual_values, classes=[0, 1, 2])

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
        '''

    # Print accuracies
    mean_acc, CI_acc_low, CI_acc_high = mean_confidence_interval(accuracies)
    print("Avg accuracy: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(mean_acc, CI_acc_low, CI_acc_high))

    # Print precision, recall, f1-score
    for i in range(n_classes):
        mean_prec, CI_prec_low, CI_prec_high = mean_confidence_interval(precisions[i])
        mean_rec, CI_rec_low, CI_rec_high = mean_confidence_interval(recalls[i])
        mean_f1, CI_f1_low, CI_f1_high = mean_confidence_interval(f1scores[i])

        print("Avg precision class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_prec, CI_prec_low,
                                                                              CI_prec_high))
        print(
            "Avg recall class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_rec, CI_rec_low, CI_rec_high))
        print(
            "Avg f1-score class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_f1, CI_f1_low, CI_f1_high))
    '''
    for i in range(n_classes):
        mean_auc, CI_low, CI_high = mean_confidence_interval(roc_avg[i])
        print("Avg AUC class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_auc, CI_low, CI_high))
    '''

        '''
        file = open(file_path, "a")
        file.write("Fold{} - Confusion Matrix\n".format(fold_n + 1))
        file.write(str(conf_matrix[0]))
        file.write("\n")
        file.write(str(conf_matrix[1]))
        file.write("\n")
        file.write(str(conf_matrix[2]))
        file.write("\n\n")

        x1 = conf_matrix[0][0]
        x2 = conf_matrix[1][0]
        x3 = conf_matrix[2][0]

        y1 = conf_matrix[0][1]
        y2 = conf_matrix[1][1]
        y3 = conf_matrix[2][1]

        z1 = conf_matrix[0][2]
        z2 = conf_matrix[1][2]
        z3 = conf_matrix[2][2]

        TP = x1 + y2 + z3
        TOT = x1 + x2 + x3 + y1 + y2 + y3 + z1 + z2 + z3

        acc = TP / TOT
        accuracies[fold_n] = acc

        TP_A = x1
        TN_A = y2 + y3 + z2 + z3
        FP_A = x2 + x3
        FN_A = y1 + z1

        TP_B = y2
        TN_B = x1 + x3 + z1 + z3
        FP_B = y1 + y3
        FN_B = x2 + z2

        TP_U = z3
        TN_U = x1 + x2 + y1 + y2
        FP_U = z1 + z2
        FN_U = x3 + y3

        rec_A = TP_A / (TP_A + FN_A)
        rec_B = TP_B / (TP_B + FN_B)
        rec_U = TP_U / (TP_U + FN_U)
        recalls[0][fold_n] = rec_A
        recalls[1][fold_n] = rec_B
        recalls[2][fold_n] = rec_U

        prec_A = TP_A / (TP_A + FP_A)
        prec_B = TP_B / (TP_B + FP_B)
        prec_U = TP_U / (TP_U + FP_U)
        precisions[0][fold_n] = prec_A
        precisions[1][fold_n] = prec_B
        precisions[2][fold_n] = prec_U

        f1_A = 2 * (prec_A * rec_A) / (prec_A + rec_A)
        f1_B = 2 * (prec_B * rec_B) / (prec_B + rec_B)
        f1_U = 2 * (prec_U * rec_U) / (prec_U + rec_U)
        f1scores[0][fold_n] = spec_A
        f1scores[1][fold_n] = spec_B
        f1scores[2][fold_n] = spec_U


    print(precisions)
    print(recalls)
    print(f1scores)
    print(accuracies)

    file.write("\nSensitivities\n")
    file.write(str(sensitivities))
    file.write("\nSpecificities\n")
    file.write(str(specificities))
    file.write("\nPrecisions\n")
    file.write(str(precisions))
    file.write("\nAccuracies\n")
    file.write(str(accuracies))

    avg_sens_A = np.mean(sensitivities[0])
    std_sens_A = np.std(sensitivities[0])
    avg_sens_B = np.mean(sensitivities[1])
    std_sens_B = np.std(sensitivities[1])
    avg_sens_U = np.mean(sensitivities[2])
    std_sens_U = np.std(sensitivities[2])

    avg_spec_A = np.mean(specificities[0])
    std_spec_A = np.std(specificities[0])
    avg_spec_B = np.mean(specificities[1])
    std_spec_B = np.std(specificities[1])
    avg_spec_U = np.mean(specificities[2])
    std_spec_U = np.std(specificities[2])

    avg_prec_A = np.mean(precisions[0])
    std_prec_A = np.std(precisions[0])
    avg_prec_B = np.mean(precisions[1])
    std_prec_B = np.std(precisions[1])
    avg_prec_U = np.mean(precisions[2])
    std_prec_U = np.std(precisions[2])



    tab = "Class\t\tSensitivity(Recall)\t\tSpecificity\t\tPrecision\n" \
          "A\t\t{:0.2f}({:0.2f})%\t\t{:0.2f}({:0.2f})%\t\t{:0.2f}({:0.2f})%\n" \
          "B\t\t{:0.2f}({:0.2f})%\t\t{:0.2f}({:0.2f})%\t\t{:0.2f}({:0.2f})%\n" \
          "U\t\t{:0.2f}({:0.2f})%\t\t{:0.2f}({:0.2f})%\t\t{:0.2f}({:0.2f})%\n".format(avg_sens_A, std_sens_A,
                                                                                      avg_spec_A, std_spec_A,
                                                                                      avg_prec_A, std_prec_A,
                                                                                      avg_sens_B, std_sens_B,
                                                                                      avg_spec_B, std_spec_B,
                                                                                      avg_prec_B, std_prec_B,
                                                                                      avg_sens_U, std_sens_U,
                                                                                      avg_spec_U, std_spec_U,
                                                                                      avg_prec_U, std_prec_U)

    print(tab)

    file.write("\n\n")
    file.write(tab)

    avg_precision = (avg_prec_A + avg_prec_B + avg_prec_U) / 3
    avg_recall = (avg_sens_A + avg_sens_B + avg_sens_U) / 3
    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    metrics = "Average precision: {:0.2f}\nAverage recall: {:0.2f}\nF1 score: {:0.2f}\nAverage accuracy: {:0.2f}({:0.2f})\n".format(
        avg_precision, avg_recall, f1_score, avg_acc, std_acc)

    print(metrics)

    file.write("\n\n")
    file.write(metrics)


    file.close()


    i = 0
    j = 0
    k = 0
    
    classic_cascade = False
    
    if classic_cascade:
    
        third_model = load_model(model_path + "Fold3_A1A2A3_notflipped-retrainAll-categorical-Inception-1569509422.model")
    
        for img_path in sorted(glob.glob(output_path_AB + "*.png"), key=os.path.getsize):
    
            img = image.load_img(img_path, target_size=(image_size, image_size))
            X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
    
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
    
            preds = third_model.predict(x)
    
            class_idx = np.argmax(preds, axis=1)
    
            if class_idx == 0:
                print("A1")
                i += 1
    
            elif class_idx == 1:
                print("A2")
                j += 1
    
            elif class_idx == 2:
                print("A3")
                k += 1
    else:
    
        third_model_A1A2 = load_model(model_path + "Fold1_A1_A2-binary-baselineInception-1569514982.model")
        third_model_A1A3 = load_model(model_path + "Fold1_A1_A3-binary-baselineInception-1569535118.model")
        third_model_A2A3 = load_model(model_path + "Fold3_A2_A3-binary-baselineInception-1569598028.model")
        third_model = load_model(model_path + "Fold3_A1A2A3_notflipped-retrainAll-categorical-Inception-1569509422.model")
    
        for img_path in sorted(glob.glob(output_path_AB + "*.png"), key=os.path.getsize):
    
            img = image.load_img(img_path, target_size=(image_size, image_size))
            X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
    
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
    
            predsA1A2 = third_model_A1A2.predict(x)  # 0 se A1, 1 se A2
            predsA1A3 = third_model_A1A3.predict(x)  # 0 se A1, 1 se A3
            predsA2A3 = third_model_A2A3.predict(x)  # 0 se A2, 1 se A3
    
            preds = third_model.predict(x)
    
            A1val = predsA1A2[0][0] + predsA1A3[0][0] + preds[0][0]
            A2val = predsA1A2[0][1] + predsA2A3[0][0] + preds[0][1]
            A3val = predsA1A3[0][1] + predsA2A3[0][1] + preds[0][2]
    
            values = [[A1val, A2val, A3val]]
    
            class_idx = np.argmax(values, axis=1)
    
            if class_idx == 0:
                print("A1")
                i += 1
    
            elif class_idx == 1:
                print("A2")
                j += 1
    
            elif class_idx == 2:
                print("A3")
                k += 1
    
    print("A1 {} - A2 {} - A3 {}".format(i, j, k))
    
    
    shutil.rmtree(output_path)
    shutil.rmtree(output_path_AB)
    os.mkdir(output_path)
    os.mkdir(output_path_AB)
    
    '''
