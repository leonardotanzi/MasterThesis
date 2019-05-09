import cv2
import os
import pandas as pd
import glob
import shutil


out_path = '/Users/leonardotanzi/Desktop/OutLabelled/'

data = pd.read_excel('/Users/leonardotanzi/Desktop/MasterThesis/DatasetCleaning/Classificazione.xlsx')
df = pd.DataFrame(data, columns=['Nome', 'Frattura', 'Lato', 'Class. AO'])


for image_path in glob.glob('/Users/leonardotanzi/Desktop/Out/*.png'):

	image_name = image_path.split('/')[-1].split('.')[0]
	found = 0

	for index, row in df.iterrows():

		if image_name == row['Nome'] and row['Frattura'] == 0:
			final_path = out_path + 'Unbroken/' + image_path.split('/')[-1]
			# print(final_path)
			found = 1
			break

		elif image_name == row['Nome'] and row['Frattura'] == 1:

			if row['Lato'] == 'Sn' or row['Lato'] == 'SN':
				side = 'Left/'
			elif row['Lato'] == 'Dx' or row['Lato'] == 'DX':
				side = 'Right/'

			if row['Class. AO'] == 'A':
				classification = 'A/'
			elif row['Class. AO'] == 'B':
				classification = 'B/'
			elif row['Class. AO'] == 'C':
				classification = 'C/'

			final_path = out_path + classification + side + image_path.split('/')[-1]
			found = 1
			break

	if found == 0:
		final_path = out_path + 'Discarded/' + image_path.split('/')[-1]

	shutil.copy(image_path, final_path)











