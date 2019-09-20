import os
import glob
from shutil import copyfile
import cv2

root_dir = "/Users/leonardotanzi/Desktop/NeededDataset/PcaCNNsmall/"

for image_path in glob.iglob(root_dir + "**/*", recursive=True):
		
	if image_path.endswith("left.png"):
		img = cv2.imread(image_path)
		flipHorizontal = cv2.flip(img, 1)
		cv2.imwrite(image_path, flipHorizontal)	

