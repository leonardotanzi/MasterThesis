from sklearn.model_selection import KFold
import numpy
from tqdm import tqdm
import cv2
import os
import numpy as np

K = 5
image_size = 256
categories = ["A", "B", "Unbroken"]

input_folders = ["/mnt/data/ltanzi/flippedDataset/{}".format(categories[0]),
                 "/mnt/data/ltanzi/flippedDataset/{}".format(categories[1]),
                 "/mnt/data/ltanzi/flippedDataset/{}".format(categories[2])]

# create the root folder
output_path = "/mnt/data/ltanzi/flippedCrossVal"
os.mkdir(output_path)

# create folders for splitting the dataset
for i in range(K):
    os.chdir(output_path)
    name_fold = "Fold{}".format(i + 1)
    os.mkdir(name_fold)
    os.chdir(output_path + "/" + name_fold)
    os.mkdir("Validation")
    os.chdir(output_path + "/" + name_fold + "/Validation")
    for cat in categories:
        os.mkdir(cat)
    os.chdir("..")
    os.mkdir("Train")
    os.chdir(output_path + "/" + name_fold + "/Train")
    for cat in categories:
        os.mkdir(cat)


for enum, path in enumerate(input_folders):

    X = []
    X_original = []
    names = []
    shapes = []

    for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
            X.append(new_array)  # add this to our training_data
            X_original.append(img_array)
            shapes.append(img_array.shape)
            names.append(img)

        except Exception as e:  # in the interest in keeping the output clean...
            pass

    X = np.array(X).reshape(-1, image_size, image_size, 1)

    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    nFold = 1
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        for i in train_index:
            cv2.imwrite(output_path + "/Fold{}/Train/{}/{}".format(nFold, categories[enum], names[i]), X_original[i])
        for i in test_index:
            cv2.imwrite(output_path + "/Fold{}/Validation/{}/{}".format(nFold, categories[enum], names[i]), X_original[i])
        nFold += 1
