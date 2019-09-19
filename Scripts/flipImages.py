import cv2
import glob
import os
from shutil import copyfile

path = "/mnt/data/ltanzi/A_flipped/"
for image_path in glob.glob(path + "*.png"):
	print(image_path)
		
	if image_path.endswith("left.png"):
		img = cv2.imread(image_path)
		flipHorizontal = cv2.flip(img, 1)
		cv2.imwrite(image_path, flipHorizontal)	



