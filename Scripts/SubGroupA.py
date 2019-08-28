import xlrd 
import cv2
import numpy as np
from shutil import copyfile
import glob


loc = "/Users/leonardotanzi/Desktop/Classificazione.xlsx"
path = "/Users/leonardotanzi/Desktop/A/"

wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
 
for file_path in glob.glob(path + "*.png"):

	sheet.cell_value(0, 0)
	full_name = file_path.split("/")[-1]
	name = file_path.split("/")[-1].split("_")[0]

	for i in range(1, sheet.nrows):
		a = sheet.row_values(i)
		if a[0] == name and (a[5] == 1.0 or a[5] == 2.0 or a[5] == 3.0):
			out_path = path + "A{}/".format(int(a[5])) + full_name
			in_path = path + full_name
			copyfile(in_path, out_path)
