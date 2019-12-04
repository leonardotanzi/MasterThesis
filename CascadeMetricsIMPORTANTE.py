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
            y_score_dict["{}".format(original_name)] = (0, 0, 0)

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
                    y_score_dict["{}".format(original_name)] = preds

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

                y_score_dict["{}".format(original_name)] = preds

            shutil.rmtree(output_path)
            shutil.rmtree(output_path_AB)
            os.mkdir(output_path)
            os.mkdir(output_path_AB)

        #PARTIRE DA QUA, VEDERE COME CONVERTIRE I DICT IN NDARRAY
        y_score_dict = {'Pelvis11_left': array([[0.6923918 , 0.30760822]], dtype=float32), 'Pelvis239_left': array([[0.99862814, 0.00137194]], dtype=float32), 'Pelvis47_right': array([[0.9764473 , 0.02355272]], dtype=float32), 'Pelvis25_right': array([[0.03091013, 0.9690899 ]], dtype=float32), 'Pelvis65_right': array([[0.99896526, 0.00103471]], dtype=float32), 'Pelvis48_right': array([[0.99534696, 0.0046531 ]], dtype=float32), 'Pelvis125_left': array([[0.99686867, 0.00313132]], dtype=float32), 'Pelvis123_left': array([[9.9945277e-01, 5.4725521e-04]], dtype=float32), 'Pelvis13_left': array([[1.3877264e-04, 9.9986124e-01]], dtype=float32), 'Pelvis1_right': array([[9.9992716e-01, 7.2850453e-05]], dtype=float32), 'Pelvis45_left': array([[0.9495333 , 0.05046674]], dtype=float32), 'Pelvis273_right': array([[0.00337735, 0.9966227 ]], dtype=float32), 'Pelvis275_left': array([[9.9998188e-01, 1.8178296e-05]], dtype=float32), 'Pelvis7_right': array([[3.3742158e-06, 9.9999666e-01]], dtype=float32), 'Pelvis30_left': array([[0.9960452 , 0.00395475]], dtype=float32), 'Pelvis268_left': array([[0.5569352 , 0.44306484]], dtype=float32), 'Pelvis237_left': array([[0.996567  , 0.00343295]], dtype=float32), 'Pelvis62_left': array([[0.9931759 , 0.00682413]], dtype=float32), 'Pelvis28_right': array([[0.8710244 , 0.12897557]], dtype=float32), 'Pelvis252_left': array([[0.01178629, 0.98821366]], dtype=float32), 'Pelvis277_left': array([[0.3202307, 0.6797693]], dtype=float32), 'Pelvis272_right': array([[0.00298549, 0.9970145 ]], dtype=float32), 'Pelvis2_right': array([[0.99158114, 0.00841882]], dtype=float32), 'Pelvis53_right': array([[9.999913e-01, 8.757029e-06]], dtype=float32), 'Pelvis127_left': array([[9.9999833e-01, 1.7121262e-06]], dtype=float32), 'Pelvis232_right': array([[0.9987488 , 0.00125123]], dtype=float32), 'Pelvis251_left': array([[0.9039988 , 0.09600119]], dtype=float32), 'Pelvis242_right': array([[0.6740815, 0.3259185]], dtype=float32), 'Pelvis68_left': array([[0.9548355 , 0.04516454]], dtype=float32), 'Pelvis33_left': array([[0.9973484 , 0.00265153]], dtype=float32), 'Pelvis14_right': array([[6.9153481e-05, 9.9993086e-01]], dtype=float32), 'Pelvis39_left': array([[9.999064e-01, 9.359967e-05]], dtype=float32), 'Pelvis255_left': array([[0.9988237 , 0.00117625]], dtype=float32), 'Pelvis70_left': array([[0.97393566, 0.02606434]], dtype=float32), 'Pelvis181_right': array([[0.99799216, 0.00200788]], dtype=float32), 'Pelvis55_right': array([[1.0000000e+00, 1.4713301e-09]], dtype=float32), 'Pelvis69_right': array([[9.9999976e-01, 2.8630885e-07]], dtype=float32), 'Pelvis265_right': array([[0.99830556, 0.00169445]], dtype=float32), 'Pelvis44_right': array([[9.9999940e-01, 6.2990716e-07]], dtype=float32), 'Pelvis46_right': array([[9.9961424e-01, 3.8567733e-04]], dtype=float32), 'Pelvis220_right': array([[0.92229474, 0.07770523]], dtype=float32), 'Pelvis27_left': array([[0.995637  , 0.00436298]], dtype=float32), 'Pelvis17_right': array([[9.999962e-01, 3.855944e-06]], dtype=float32), 'Pelvis244_right': array([[9.992235e-01, 7.764883e-04]], dtype=float32), 'Pelvis5_right': array([[0.99873716, 0.00126283]], dtype=float32), 'Pelvis42_left': array([[0.9982324 , 0.00176758]], dtype=float32), 'Pelvis8_right': array([[1.0000000e+00, 2.0354587e-09]], dtype=float32), 'Pelvis58_left': array([[0.95301646, 0.04698359]], dtype=float32), 'Pelvis126_right': array([[0.01331787, 0.9866822 ]], dtype=float32), 'Pelvis15_right': array([[0.53235006, 0.4676499 ]], dtype=float32), 'Pelvis57_right': array([[9.9999809e-01, 1.8920949e-06]], dtype=float32), 'Pelvis210_left': array([[0.96182823, 0.03817175]], dtype=float32), 'Pelvis66_right': array([[1.0000000e+00, 2.3628326e-08]], dtype=float32), 'Pelvis274_left': array([[9.996973e-01, 3.026610e-04]], dtype=float32), 'Pelvis60_left': array([[9.997032e-01, 2.967935e-04]], dtype=float32), 'Pelvis266_right': array([[9.9983728e-01, 1.6274898e-04]], dtype=float32), 'Pelvis34_right': array([[9.999777e-01, 2.231375e-05]], dtype=float32), 'Pelvis119_right': array([[0.90687877, 0.09312121]], dtype=float32), 'Pelvis203_right': array([[0.82678133, 0.17321861]], dtype=float32), 'Pelvis257_left': array([[0.43586287, 0.5641371 ]], dtype=float32), 'Pelvis31_right': array([[9.9996185e-01, 3.8167334e-05]], dtype=float32), 'Pelvis50_right': array([[0.97339904, 0.02660095]], dtype=float32), 'Pelvis218_left': array([[0.03675498, 0.96324503]], dtype=float32), 'Pelvis71_left': array([[0.8786063 , 0.12139374]], dtype=float32), 'Pelvis56_right': array([[0.9802675, 0.0197325]], dtype=float32), 'Pelvis43_right': array([[9.999913e-01, 8.748615e-06]], dtype=float32), 'Pelvis10_right': array([[0.89057904, 0.10942091]], dtype=float32), 'Pelvis103_right': array([[0.16770546, 0.8322945 ]], dtype=float32), 'Pelvis245_left': array([[9.998128e-01, 1.871910e-04]], dtype=float32), 'Pelvis64_right': array([[0.99418133, 0.00581863]], dtype=float32), 'Pelvis267_left': array([[9.993432e-01, 6.567539e-04]], dtype=float32), 'Pelvis217_left': array([[0.06675062, 0.93324935]], dtype=float32), 'Pelvis24_right': array([[0.99226403, 0.00773596]], dtype=float32), 'Pelvis54_left': array([[9.991763e-01, 8.236387e-04]], dtype=float32), 'Pelvis37_left': array([[0.86319876, 0.13680118]], dtype=float32), 'Pelvis9_left': array([[9.9948597e-01, 5.1398267e-04]], dtype=float32), 'Pelvis264_right': array([[0.99873   , 0.00126995]], dtype=float32), 'Pelvis198_left': array([[0.88635707, 0.11364292]], dtype=float32), 'Pelvis59_right': array([[1.000000e+00, 3.230373e-08]], dtype=float32), 'Pelvis35_left': array([[9.999939e-01, 6.054309e-06]], dtype=float32), 'Pelvis23_right': array([[9.9997437e-01, 2.5599398e-05]], dtype=float32), 'Pelvis270_right': array([[0.78513175, 0.21486823]], dtype=float32), 'Pelvis271_right': array([[0.62264776, 0.3773522 ]], dtype=float32), 'Pelvis16_right': array([[9.9998367e-01, 1.6320218e-05]], dtype=float32), 'Pelvis18_left': array([[0.89612114, 0.10387886]], dtype=float32), 'Pelvis209_left': array([[9.9999964e-01, 3.9340853e-07]], dtype=float32), 'Pelvis52_left': array([[0.89635795, 0.10364206]], dtype=float32), 'Pelvis78_left': array([[0.7412728 , 0.25872722]], dtype=float32), 'Pelvis26_right': array([[1.0000000e+00, 1.7488896e-11]], dtype=float32), 'Pelvis216_left': array([[0.00829313, 0.99170685]], dtype=float32), 'Pelvis211_left': array([[0.00295067, 0.99704933]], dtype=float32), 'Pelvis29_left': array([[9.999379e-01, 6.205724e-05]], dtype=float32), 'Pelvis250_left': array([[0.4764177, 0.5235823]], dtype=float32), 'Pelvis199_left': array([[0.1911219, 0.8088781]], dtype=float32), 'Pelvis249_right': array([[9.993717e-01, 6.282664e-04]], dtype=float32), 'Pelvis248_right': array([[9.999645e-01, 3.546908e-05]], dtype=float32), 'Pelvis120_right': array([[9.9997795e-01, 2.2052909e-05]], dtype=float32), 'Pelvis121_right': array([[9.9925762e-01, 7.4239576e-04]], dtype=float32), 'Pelvis22_right': array([[9.9999809e-01, 1.9477138e-06]], dtype=float32), 'Pelvis20_right': array([[9.9999583e-01, 4.1205767e-06]], dtype=float32), 'Pelvis153_right': array([[8.416109e-08, 9.999999e-01]], dtype=float32), 'Pelvis130_right': array([[2.4329606e-06, 9.9999762e-01]], dtype=float32), 'Pelvis154_right': array([[3.9546622e-04, 9.9960452e-01]], dtype=float32), 'Pelvis162_right': array([[3.6482295e-05, 9.9996352e-01]], dtype=float32), 'Pelvis188_right': array([[2.6056057e-04, 9.9973947e-01]], dtype=float32), 'Pelvis172_left': array([[0.6461548 , 0.35384518]], dtype=float32), 'Pelvis157_right': array([[3.1677436e-04, 9.9968326e-01]], dtype=float32), 'Pelvis160_left': array([[7.3785395e-07, 9.9999928e-01]], dtype=float32), 'Pelvis147_left': array([[4.979199e-06, 9.999950e-01]], dtype=float32), 'Pelvis224_left': array([[1.9040854e-06, 9.9999809e-01]], dtype=float32), 'Pelvis118_left': array([[0.56901085, 0.43098912]], dtype=float32), 'Pelvis231_right': array([[2.0470939e-07, 9.9999976e-01]], dtype=float32), 'Pelvis262_right': array([[0.00842365, 0.9915764 ]], dtype=float32), 'Pelvis166_right': array([[4.2762144e-06, 9.9999571e-01]], dtype=float32), 'Pelvis174_right': array([[0.0034407, 0.9965593]], dtype=float32), 'Pelvis192_right': array([[0.0704348, 0.9295652]], dtype=float32), 'Pelvis182_right': array([[0.00414518, 0.9958548 ]], dtype=float32), 'Pelvis247_right': array([[3.0321578e-06, 9.9999702e-01]], dtype=float32), 'Pelvis175_left': array([[8.906382e-04, 9.991093e-01]], dtype=float32), 'Pelvis163_right': array([[3.6239810e-05, 9.9996376e-01]], dtype=float32), 'Pelvis161_left': array([[6.151163e-06, 9.999938e-01]], dtype=float32), 'Pelvis138_right': array([[3.805568e-05, 9.999620e-01]], dtype=float32), 'Pelvis201_left': array([[0.48961228, 0.5103878 ]], dtype=float32), 'Pelvis169_left': array([[2.5283492e-07, 9.9999976e-01]], dtype=float32), 'Pelvis137_right': array([[0.0016946 , 0.99830544]], dtype=float32), 'Pelvis144_right': array([[1.8039422e-06, 9.9999821e-01]], dtype=float32), 'Pelvis165_left': array([[5.0705135e-06, 9.9999487e-01]], dtype=float32), 'Pelvis170_left': array([[5.3676456e-04, 9.9946326e-01]], dtype=float32), 'Pelvis131_left': array([[3.7094317e-06, 9.9999630e-01]], dtype=float32), 'Pelvis193_right': array([[6.3464454e-06, 9.9999368e-01]], dtype=float32), 'Pelvis200_left': array([[0.06301581, 0.93698424]], dtype=float32), 'Pelvis155_right': array([[3.223406e-08, 1.000000e+00]], dtype=float32), 'Pelvis235_left': array([[3.2041644e-05, 9.9996793e-01]], dtype=float32), 'Pelvis167_left': array([[0.28195992, 0.71804005]], dtype=float32), 'Pelvis124_right': array([[0.01043326, 0.98956674]], dtype=float32), 'Pelvis208_right': array([[0.00133794, 0.9986621 ]], dtype=float32), 'Pelvis206_left': array([[4.435272e-06, 9.999956e-01]], dtype=float32), 'Pelvis134_right': array([[1.3020719e-05, 9.9998701e-01]], dtype=float32), 'Pelvis122_right': array([[0.00481055, 0.9951894 ]], dtype=float32), 'Pelvis148_left': array([[1.4939842e-06, 9.9999845e-01]], dtype=float32), 'Pelvis164_left': array([[9.543984e-05, 9.999045e-01]], dtype=float32), 'Pelvis194_left': array([[0.62136436, 0.37863567]], dtype=float32), 'Pelvis180_left': array([[0.00214701, 0.99785304]], dtype=float32), 'Pelvis236_left': array([[3.1273110e-05, 9.9996877e-01]], dtype=float32), 'Pelvis135_right': array([[1.8496223e-06, 9.9999809e-01]], dtype=float32), 'Pelvis207_right': array([[0.01423177, 0.98576826]], dtype=float32), 'Pelvis146_right': array([[1.4220279e-04, 9.9985778e-01]], dtype=float32), 'Pelvis185_left': array([[3.2565025e-05, 9.9996746e-01]], dtype=float32), 'Pelvis213_left': array([[6.289335e-07, 9.999994e-01]], dtype=float32), 'Pelvis234_left': array([[1.0240098e-05, 9.9998975e-01]], dtype=float32), 'Pelvis215_right': array([[0.993341, 0.006659]], dtype=float32), 'Pelvis107_right': array([[0.2630591, 0.7369409]], dtype=float32), 'Pelvis156_right': array([[3.5809023e-06, 9.9999642e-01]], dtype=float32), 'Pelvis177_right': array([[2.434676e-06, 9.999976e-01]], dtype=float32), 'Pelvis176_right': array([[2.1101629e-07, 9.9999976e-01]], dtype=float32), 'Pelvis204_right': array([[4.8173613e-05, 9.9995184e-01]], dtype=float32), 'Pelvis178_left': array([[8.012368e-05, 9.999199e-01]], dtype=float32), 'Pelvis256_right': array([[7.820546e-05, 9.999218e-01]], dtype=float32), 'Pelvis260_right': array([[6.2609455e-05, 9.9993742e-01]], dtype=float32), 'Pelvis186_left': array([[2.2017118e-04, 9.9977988e-01]], dtype=float32), 'Pelvis233_left': array([[5.2006867e-06, 9.9999475e-01]], dtype=float32), 'Pelvis152_right': array([[9.5925250e-05, 9.9990404e-01]], dtype=float32), 'Pelvis100_left': array([[0.33494547, 0.6650545 ]], dtype=float32), 'Pelvis143_right': array([[0.1159264 , 0.88407356]], dtype=float32), 'Pelvis214_left': array([[0.96013784, 0.03986222]], dtype=float32), 'Pelvis151_right': array([[3.7788646e-05, 9.9996221e-01]], dtype=float32), 'Pelvis222_right': array([[1.4034390e-06, 9.9999857e-01]], dtype=float32), 'Pelvis261_right': array([[1.7365428e-05, 9.9998260e-01]], dtype=float32), 'Pelvis173_right': array([[5.5541179e-05, 9.9994445e-01]], dtype=float32), 'Pelvis133_left': array([[2.1509244e-07, 9.9999976e-01]], dtype=float32), 'Pelvis219_right': array([[0.9737101 , 0.02628986]], dtype=float32), 'Pelvis141_right': array([[0.0070593 , 0.99294066]], dtype=float32), 'Pelvis259_right': array([[0.98545766, 0.01454226]], dtype=float32), 'Pelvis258_right': array([[0.7206868 , 0.27931327]], dtype=float32), 'Pelvis89_left': array([[0.28608292, 0.713917  ]], dtype=float32), 'Pelvis240_left': array([[0.3375057 , 0.66249436]], dtype=float32), 'Pelvis183_right': array([[4.903263e-05, 9.999510e-01]], dtype=float32), 'Pelvis189_left': array([[1.4391626e-04, 9.9985611e-01]], dtype=float32), 'Pelvis230_right': array([[3.1146652e-07, 9.9999964e-01]], dtype=float32), 'Pelvis136_right': array([[0.16548383, 0.83451617]], dtype=float32), 'Pelvis223_left': array([[5.1104306e-04, 9.9948895e-01]], dtype=float32), 'Pelvis140_right': array([[6.5441994e-04, 9.9934560e-01]], dtype=float32), 'Pelvis212_right': array([[0.08039963, 0.9196003 ]], dtype=float32), 'Pelvis226_left': array([[0.39494765, 0.60505235]], dtype=float32), 'Pelvis187_left': array([[1.4796768e-05, 9.9998522e-01]], dtype=float32), 'Pelvis132_left': array([[1.3702876e-07, 9.9999988e-01]], dtype=float32), 'Pelvis202_right': array([[0.00303254, 0.9969675 ]], dtype=float32), 'Pelvis221_left': array([[2.1924899e-04, 9.9978071e-01]], dtype=float32), 'Pelvis158_left': array([[0.46487707, 0.5351229 ]], dtype=float32), 'Pelvis159_left': array([[1.2811895e-04, 9.9987185e-01]], dtype=float32), 'Pelvis94_right': array([[0.00509025, 0.9949097 ]], dtype=float32), 'Pelvis263_right': array([[8.750117e-05, 9.999125e-01]], dtype=float32), 'Pelvis225_left': array([[2.2597950e-04, 9.9977404e-01]], dtype=float32), 'Pelvis228_right': array([[8.8597875e-04, 9.9911398e-01]], dtype=float32), 'Pelvis205_right': array([[3.786389e-05, 9.999621e-01]], dtype=float32), 'Pelvis149_left': array([[0.00138291, 0.99861705]], dtype=float32), 'Pelvis150_left': array([[1.4983733e-06, 9.9999845e-01]], dtype=float32), 'Pelvis227_left': array([[3.3716426e-06, 9.9999666e-01]], dtype=float32), 'Pelvis190_right': array([[2.3203293e-07, 9.9999976e-01]], dtype=float32), 'Pelvis191_right': array([[1.0935582e-04, 9.9989069e-01]], dtype=float32), 'Pelvis114_right': array([[4.1644502e-04, 9.9958354e-01]], dtype=float32), 'Pelvis114_left': array([[0.00368204, 0.996318  ]], dtype=float32), 'Pelvis85_left': array([[1.9470788e-06, 9.9999809e-01]], dtype=float32), 'Pelvis75_left': array([[7.3149183e-04, 9.9926847e-01]], dtype=float32), 'Pelvis115_right': array([[0.14263469, 0.8573653 ]], dtype=float32), 'Pelvis116_left': array([[6.9810230e-05, 9.9993014e-01]], dtype=float32), 'Pelvis116_right': array([[0.00214817, 0.99785185]], dtype=float32), 'Pelvis85_right': array([[0.06627252, 0.9337275 ]], dtype=float32), 'Pelvis81_right': array([[4.1632456e-10, 1.0000000e+00]], dtype=float32), 'Pelvis48_left': array([[2.6542786e-05, 9.9997342e-01]], dtype=float32), 'Pelvis81_left': array([[1.3501685e-04, 9.9986494e-01]], dtype=float32), 'Pelvis47_left': array([[0.00269416, 0.99730587]], dtype=float32), 'Pelvis11_right': array([[9.5102041e-07, 9.9999905e-01]], dtype=float32), 'Pelvis61_left': array([[2.3132641e-04, 9.9976867e-01]], dtype=float32), 'Pelvis27_right': array([[2.044303e-04, 9.997956e-01]], dtype=float32), 'Pelvis79_right': array([[0.00107172, 0.99892825]], dtype=float32), 'Pelvis77_right': array([[2.5997215e-06, 9.9999738e-01]], dtype=float32), 'Pelvis83_left': array([[1.4088336e-04, 9.9985909e-01]], dtype=float32), 'Pelvis55_left': array([[1.17341195e-08, 1.00000000e+00]], dtype=float32), 'Pelvis69_left': array([[2.9815194e-05, 9.9997020e-01]], dtype=float32), 'Pelvis65_left': array([[4.8162356e-06, 9.9999523e-01]], dtype=float32), 'Pelvis73_right': array([[4.6603827e-07, 9.9999952e-01]], dtype=float32), 'Pelvis117_left': array([[4.2681317e-05, 9.9995732e-01]], dtype=float32), 'Pelvis99_right': array([[1.7986365e-05, 9.9998200e-01]], dtype=float32), 'Pelvis54_right': array([[3.2501067e-08, 1.0000000e+00]], dtype=float32), 'Pelvis72_left': array([[0.00885864, 0.9911413 ]], dtype=float32), 'Pelvis98_left': array([[0.00487194, 0.99512804]], dtype=float32), 'Pelvis39_right': array([[7.772029e-05, 9.999223e-01]], dtype=float32), 'Pelvis79_left': array([[1.0991546e-06, 9.9999893e-01]], dtype=float32), 'Pelvis44_left': array([[0.04348208, 0.95651793]], dtype=float32), 'Pelvis83_right': array([[1.2241377e-06, 9.9999881e-01]], dtype=float32), 'Pelvis104_right': array([[1.257610e-06, 9.999987e-01]], dtype=float32), 'Pelvis108_left': array([[2.1364725e-05, 9.9997866e-01]], dtype=float32), 'Pelvis70_right': array([[5.2759873e-07, 9.9999952e-01]], dtype=float32), 'Pelvis92_right': array([[2.3271689e-06, 9.9999762e-01]], dtype=float32), 'Pelvis109_left': array([[7.4202865e-07, 9.9999928e-01]], dtype=float32), 'Pelvis10_left': array([[0.02790774, 0.9720923 ]], dtype=float32), 'Pelvis98_right': array([[2.1935411e-06, 9.9999785e-01]], dtype=float32), 'Pelvis40_right': array([[0.0257844 , 0.97421557]], dtype=float32), 'Pelvis104_left': array([[6.815784e-11, 1.000000e+00]], dtype=float32), 'Pelvis77_left': array([[2.3173772e-05, 9.9997687e-01]], dtype=float32), 'Pelvis76_right': array([[1.1629885e-06, 9.9999881e-01]], dtype=float32), 'Pelvis105_right': array([[0.00244273, 0.9975573 ]], dtype=float32), 'Pelvis26_left': array([[1.4212546e-05, 9.9998581e-01]], dtype=float32), 'Pelvis51_left': array([[3.7674513e-07, 9.9999964e-01]], dtype=float32), 'Pelvis53_left': array([[0.2751198, 0.7248802]], dtype=float32), 'Pelvis52_right': array([[0.0019421, 0.9980579]], dtype=float32), 'Pelvis57_left': array([[0.25197268, 0.7480273 ]], dtype=float32), 'Pelvis9_right': array([[0.00113031, 0.9988697 ]], dtype=float32), 'Pelvis118_right': array([[1.4275486e-08, 1.0000000e+00]], dtype=float32), 'Pelvis62_right': array([[3.1527637e-05, 9.9996853e-01]], dtype=float32), 'Pelvis13_right': array([[9.997539e-01, 2.461338e-04]], dtype=float32), 'Pelvis45_right': array([[0.00157699, 0.998423  ]], dtype=float32), 'Pelvis92_left': array([[6.035798e-08, 9.999999e-01]], dtype=float32), 'Pelvis71_right': array([[0.0018129, 0.9981871]], dtype=float32), 'Pelvis23_left': array([[3.8551945e-05, 9.9996150e-01]], dtype=float32), 'Pelvis63_left': array([[3.923944e-04, 9.996076e-01]], dtype=float32), 'Pelvis33_right': array([[2.1466170e-07, 9.9999976e-01]], dtype=float32), 'Pelvis60_right': array([[0.07811996, 0.92188007]], dtype=float32), 'Pelvis17_left': array([[3.6035211e-12, 1.0000000e+00]], dtype=float32), 'Pelvis28_left': array([[3.5537762e-04, 9.9964464e-01]], dtype=float32), 'Pelvis96_left': array([[1.6738963e-11, 1.0000000e+00]], dtype=float32), 'Pelvis113_right': array([[5.0619984e-04, 9.9949384e-01]], dtype=float32), 'Pelvis112_right': array([[0.01272153, 0.98727846]], dtype=float32), 'Pelvis41_right': array([[1.7437876e-10, 1.0000000e+00]], dtype=float32), 'Pelvis106_right': array([[4.6305945e-06, 9.9999535e-01]], dtype=float32), 'Pelvis58_right': array([[1.6745811e-06, 9.9999833e-01]], dtype=float32), 'Pelvis108_right': array([[2.9858069e-07, 9.9999964e-01]], dtype=float32), 'Pelvis76_left': array([[3.532208e-05, 9.999647e-01]], dtype=float32), 'Pelvis112_left': array([[0.01969733, 0.9803027 ]], dtype=float32), 'Pelvis8_left': array([[3.3409977e-09, 1.0000000e+00]], dtype=float32), 'Pelvis49_left': array([[1.4225351e-04, 9.9985778e-01]], dtype=float32), 'Pelvis12_left': array([[0.00250842, 0.9974916 ]], dtype=float32), 'Pelvis68_right': array([[4.754728e-04, 9.995246e-01]], dtype=float32), 'Pelvis67_left': array([[0.00803763, 0.9919624 ]], dtype=float32), 'Pelvis106_left': array([[2.8740741e-08, 1.0000000e+00]], dtype=float32), 'Pelvis110_left': array([[2.174165e-04, 9.997826e-01]], dtype=float32), 'Pelvis38_right': array([[0.10401567, 0.89598435]], dtype=float32), 'Pelvis14_left': array([[9.9990427e-01, 9.5744304e-05]], dtype=float32), 'Pelvis56_left': array([[1.2122265e-05, 9.9998784e-01]], dtype=float32), 'Pelvis64_left': array([[0.00240796, 0.9975921 ]], dtype=float32), 'Pelvis107_left': array([[5.0304731e-04, 9.9949694e-01]], dtype=float32), 'Pelvis99_left': array([[0.00487495, 0.9951251 ]], dtype=float32), 'Pelvis15_left': array([[0.00320517, 0.9967949 ]], dtype=float32), 'Pelvis100_right': array([[0.00168581, 0.99831426]], dtype=float32), 'Pelvis18_right': array([[3.7225436e-07, 9.9999964e-01]], dtype=float32), 'Pelvis113_left': array([[0.11967673, 0.88032323]], dtype=float32), 'Pelvis111_right': array([[3.351890e-05, 9.999665e-01]], dtype=float32), 'Pelvis42_right': array([[9.7006841e-06, 9.9999034e-01]], dtype=float32), 'Pelvis59_left': array([[3.6556784e-07, 9.9999964e-01]], dtype=float32), 'Pelvis35_right': array([[1.2293339e-04, 9.9987710e-01]], dtype=float32), 'Pelvis103_left': array([[0.00104506, 0.9989549 ]], dtype=float32), 'Pelvis34_left': array([[2.2754944e-06, 9.9999774e-01]], dtype=float32), 'Pelvis119_left': array([[3.7640723e-04, 9.9962354e-01]], dtype=float32), 'Pelvis32_right': array([[2.1719107e-09, 1.0000000e+00]], dtype=float32), 'Pelvis31_left': array([[0.00101022, 0.99898976]], dtype=float32), 'Pelvis37_right': array([[7.6564166e-10, 1.0000000e+00]], dtype=float32), 'Pelvis29_right': array([[1.4822547e-07, 9.9999988e-01]], dtype=float32), 'Pelvis20_left': array([[0.20089363, 0.79910636]], dtype=float32), 'Pelvis22_left': array([[0.00454381, 0.9954562 ]], dtype=float32)}
        y_score = y_score_dict.values()
        y_score = np.asarray(y_score)

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

    for i in range(n_classes):
        mean_auc, CI_low, CI_high = mean_confidence_interval(roc_avg[i])
        print("Avg AUC class {}: {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(classes[i], mean_auc, CI_low, CI_high))


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
