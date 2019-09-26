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
import xlwt
from xlwt import Workbook


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

# class1 = "Broken"
# class2 = "Unbroken"

subclass1 = "A"
subclass2 = "B"
subclass3 = "Unbroken"

wb = Workbook()

sheet = wb.add_sheet('Predictions')

sheet.write(0, 0, "Name")
sheet.write(0, 1, "Broken Probability")
sheet.write(0, 2, "Unbroken Probability")
sheet.write(0, 3, "A Probability")
sheet.write(0, 4, "B Probability")
sheet.write(0, 5, "Prediction")


if run_on_server == "y":
    score_folder = "/mnt/Data/ltanzi/Train_Val/Test"
    model_path = "/mnt/Data/ltanzi/"
    test_folder = ["/mnt/data/ltanzi/Train_Val/Testing/Test" + class1, "/mnt/data/ltanzi/Train_Val/Testing/Test" + class2]


elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/"
    score_folder = "/Users/leonardotanzi/Desktop/TestBiomeccanica/Original"
    # score_folder_A1A2A3 = "/Users/leonardotanzi/Desktop/SubgroupA_Proportioned/Test/A3"
    test_folder = ["/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass1,
                   "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass2,
                   "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Testing/Test" + subclass3]
else:
    raise ValueError("Incorrect arg.")


output_path = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/OutputBroUnbro/"
output_path_AB = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/OutputAB/"


classmode = "sparse"
image_size = 299

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

first_model = load_model(model_path + "Fold1_IncV3-Broken_Unbroken-categorical-baselineInception-1568367921-best_model.h5")
second_model = load_model(model_path + "Fold4_IncV3-A_B-categorical-baselineInception-1568304568-best_model.h5")

num_row = 1

for img_path in sorted(glob.glob(score_folder + "/*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(image_size, image_size))
    X_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to array
    original_name = img_path.split("/")[-1].split(".")[0]
    sheet.write(num_row, 0, original_name)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = first_model.predict(x)

    class_idx = np.argmax(preds, axis=1)

    sheet.write(num_row, 1, "{0:6.4f}".format(preds[0][0]))  # perchè preds[0] si riferisce ad Broken
    sheet.write(num_row, 2, "{0:6.4f}".format(preds[0][1]))  # perchè preds[1] si riferisce ad Unbroken

    if class_idx == 1:
        print("Unbroken")
        sheet.write(num_row, 5, "Unbroken")

    elif class_idx == 0:
        print("Broken")

        preds2 = second_model.predict(x)

        class_idx = np.argmax(preds2, axis=1)

        sheet.write(num_row, 3, "{0:6.4f}".format(preds2[0][0]))  # perchè preds[0] si riferisce ad A
        sheet.write(num_row, 4, "{0:6.4f}".format(preds2[0][1]))  # perchè preds[1] si riferisce a B

        if class_idx == 0:
            #cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/APredicted/{}-PredictedA.png".format(original_name),X_original)
            sheet.write(num_row, 5, "A")

        elif class_idx == 1:
            #cv2.imwrite("/Users/leonardotanzi/Desktop/NeededDataset/Cascade/BPredicted/{}-PredictedB.png".format(original_name),X_original)
            sheet.write(num_row, 5, "B")

    num_row += 1

wb.save('/Users/leonardotanzi/Desktop/Prediction.xls')
