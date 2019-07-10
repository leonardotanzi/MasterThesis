from PIL import Image
import os
import sys

directory = "/Users/leonardotanzi/Desktop/OutLabelled/Unbroken"

for file_name in os.listdir(directory):
	if file_name.endswith(".png"):
		print("Processing %s" % file_name)
		image = Image.open(os.path.join(directory, file_name))

		x,y = image.size
		new_dimensions = (900, 900)
		output = image.resize(new_dimensions, Image.ANTIALIAS)

		output_file_name = os.path.join(directory, "small_" + file_name)
		output.save(output_file_name, "JPEG", quality = 95)

print("All done")