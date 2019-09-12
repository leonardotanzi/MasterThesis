import os
import glob
from shutil import copyfile
import cv2

root_dir = "/Users/leonardotanzi/Desktop/SubgroupA_canc/"

cannyWindow = 17

for filename in glob.iglob(root_dir + "**/*", recursive=True):
	print(filename)
	print(filename.split("/")[-1].split(".")[-1])
	if filename.split("/")[-1].split(".")[-1] == "png":
		img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # convert to array
		edged_array = cv2.Canny(img_array, cannyWindow, cannyWindow * 3, apertureSize=3)
		cv2.imwrite(filename + "a.png", edged_array)

