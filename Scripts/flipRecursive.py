import os
import glob
from shutil import copyfile
import cv2

root_dir = "/mnt/data/ltanzi/A1A2A3onefoldFlipped/"
for image_path in glob.iglob(root_dir + "**/*", recursive=True):
        if image_path.endswith("left.png"):
                print(image_path)
                img = cv2.imread(image_path)
                flipHorizontal = cv2.flip(img, 1)
                cv2.imwrite(image_path, flipHorizontal)	

