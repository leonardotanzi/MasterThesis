from sklearn.model_selection import KFold
import numpy
from tqdm import tqdm
import cv2
import os
import numpy as np

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
image_size = 256

path = "/Users/leonardotanzi/Desktop/FinalDataset/Adacanc"

X = []

for img in tqdm(os.listdir(path)):  # iterate over each image per broken and unbroken
	try:
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
		new_array = cv2.resize(img_array, (image_size, image_size))  # resize to normalize data size
		X.append(new_array)  # add this to our training_data
	except Exception as e:  # in the interest in keeping the output clean...
		pass

X = np.array(X).reshape(-1, image_size, image_size, 1)

# X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# X = np.array(X)


numFold = 5
totIm = X.shape[0]
im_per_fold = int(totIm / numFold)
tot_int = im_per_fold * numFold
remain = totIm - tot_int

a = []
for i in range(remain):
	a.append(X[-1])
	X = X[:-1]

a = np.array(a)

f1 = [[],[],[],[],[]]

f1 = np.split(X, numFold)

for i, elem in enumerate(a):
	f1[i] = np.append(f1[i], elem)


for i in range(numFold):
	val = f1[i]
	train = []
	for j in range(numFold):
		if j != i:
			train.append(f1[j])
	train = np.array(train)
	train = train.flatten('F')

	for img in val:
		cv2.imwrite('messigray.png', img)