import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import argparse
import numpy as np
import glob
import cv2
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

class1 = "A"
class2 = "B"
class3 = "Unbroken"
classes = [class1, class2, class3]

# in order to avg the values for each class and fold
sensitivities = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
specificities = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
precisions = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
accuracies = [0, 0, 0, 0, 0]


if run_on_server == "y":
    model_path = "/mnt/Data/ltanzi/flipCascade/BestModels/"
    test_folder = "/mnt/data/ltanzi/flippedCrossVal/Test/"
    output_path = "/mnt/data/ltanzi/flipCascade/OutputBroUnbro/"
    output_path_AB = "/mnt/data/ltanzi/flipCascade/OutputAB/"
    file_path = "/mnt/data/ltanzi/flipCascade/metricsBest.txt"

elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/"
    test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Test/"
    output_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/OutputBroUnbro/"
    output_path_AB = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/OutputAB/"
    file_path = "/Users/leonardotanzi/Desktop/metrics.txt"

    # score_folder_A1A2A3 = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Proportioned/Test/A3"
    # test_folder_A1A2A3 = ["/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass1,
    #               "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass2,
    #               "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass3]
else:
    raise ValueError("Incorrect arg.")


image_size = 299

for fold_n in range(5):
    first_model = load_model(model_path + "Fold{}_BroUnbro-best_model.h5".format(fold_n + 1))
    second_model = load_model(model_path + "Fold{}_AB-best_model.h5".format(fold_n + 1))

    # first_model = load_model("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Fold1_IncV3-Broken_Unbroken-categorical-baselineInception-1568367921-best_model.h5")
    # second_model = load_model("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Fold4_IncV3-A_B-categorical-baselineInception-1568304568-best_model.h5")

    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for class_n in range(3):
        for img_path in sorted(glob.glob(test_folder + "{}/*.png".format(classes[class_n])), key=os.path.getsize):

            img = image.load_img(img_path, target_size=(image_size, image_size))
            X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
            original_name = img_path.split("/")[-1].split(".")[0]

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = first_model.predict(x)

            class_idx = np.argmax(preds, axis=1)

            if class_idx == 1:
                print("Unbroken")
                confusion_matrix[class_n][2] += 1

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
                confusion_matrix[class_n][0] += 1
                # cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/APredicted/{}-Label{}-PredictedA.png".format(original_name, label), X_original)
                name_out = output_path_AB + "{}".format(img_path.split("/")[-1])
                cv2.imwrite(name_out, X_original)

            elif class_idx == 1:
                print("B")
                confusion_matrix[class_n][1] += 1
                # cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/BPredicted/{}-Label{}-PredictedB.png".format(original_name, label), X_original)

        shutil.rmtree(output_path)
        shutil.rmtree(output_path_AB)
        os.mkdir(output_path)
        os.mkdir(output_path_AB)


    print(confusion_matrix)
    file = open(file_path, "a")
    file.write("Fold{} - Confusion Matrix\n".format(fold_n + 1))
    file.write(str(confusion_matrix[0]))
    file.write("\n")
    file.write(str(confusion_matrix[1]))
    file.write("\n")
    file.write(str(confusion_matrix[2]))
    file.write("\n\n")

    '''
    x1 y1 z1
    x2 y2 z2
    x3 y3 z3
    '''

    x1 = confusion_matrix[0][0]
    x2 = confusion_matrix[1][0]
    x3 = confusion_matrix[2][0]

    y1 = confusion_matrix[0][1]
    y2 = confusion_matrix[1][1]
    y3 = confusion_matrix[2][1]

    z1 = confusion_matrix[0][2]
    z2 = confusion_matrix[1][2]
    z3 = confusion_matrix[2][2]

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

    sens_A = TP_A / (TP_A + FN_A)
    spec_A = TN_A / (TN_A + FP_A)
    prec_A = TP_A / (TP_A + FP_A)

    sens_B = TP_B / (TP_B + FN_B)
    spec_B = TN_B / (TN_B + FP_B)
    prec_B = TP_B / (TP_B + FP_B)

    sens_U = TP_U / (TP_U + FN_U)
    spec_U = TN_U / (TN_U + FP_U)
    prec_U = TP_U / (TP_U + FP_U)

    sensitivities[0][fold_n] = sens_A
    sensitivities[1][fold_n] = sens_B
    sensitivities[2][fold_n] = sens_U

    specificities[0][fold_n] = spec_A
    specificities[1][fold_n] = spec_B
    specificities[2][fold_n] = spec_U

    precisions[0][fold_n] = prec_A
    precisions[1][fold_n] = prec_B
    precisions[2][fold_n] = prec_U



print(sensitivities)
print(specificities)
print(precisions)
print(accuracies)

'''
Matrix format:

sensA-f1 sensA-f2 sensA-f3 sensA-f4 sensA-f5
sensB-f1 sensB-f2 sensB-f3 sensB-f4 sensB-f5
sensU-f1 sensU-f2 sensU-f3 sensU-f4 sensU-f5

'''
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

'''
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
