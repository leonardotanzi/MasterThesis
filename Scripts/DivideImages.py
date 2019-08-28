import xlrd 
import cv2
import numpy as np
from shutil import copyfile


loc = ("/Users/leonardotanzi/Desktop/Classificazione.xlsx") 
root = "/Users/leonardotanzi/Desktop/OriginalDataset/"

wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
  
sheet.cell_value(0, 0) 

for i in range(1682, sheet.nrows):

	a = sheet.row_values(i)
	in_path = root + a[0] + ".png"

	print(in_path)
	image = cv2.imread(in_path)
	height, width = image.shape[:2]
	start_row, start_col = int(0), int(0)
	# Let's get the ending pixel coordinates (bottom right of cropped top)
	end_row, end_col = int(height), int(width * 0.5)
	cropped_left = image[start_row:end_row, start_col:end_col]

	start_row, start_col = int(0), int(width * 0.5)
	# Let's get the ending pixel coordinates (bottom right of cropped bottom)
	end_row, end_col = int(height), int(width)
	cropped_right = image[start_row:end_row, start_col:end_col]

	if a[4] == "A" or a[4] == "B" or a[4] == "C":

		out_path_fracture = root + "{}/".format(a[4]) + a[0] + ".png"
		out_path = root + "Unbroken/" + a[0] + ".png"
		if a[2] == "Sn" or a[2] == "SN":
			cv2.imwrite(out_path_fracture, cropped_right)
			cv2.imwrite(out_path, cropped_left)
		elif a[2] == "Dx" or a[2] == "DX":
			cv2.imwrite(out_path_fracture, cropped_left)
			cv2.imwrite(out_path, cropped_right)

	else:
		out_path = root + "Unbroken/" + a[0] + ".png"
		cv2.imwrite(out_path, cropped_left)
		cv2.imwrite(out_path, cropped_right)
