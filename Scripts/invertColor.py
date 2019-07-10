
from PIL import Image
import PIL.ImageOps
import glob

out_path = '/Users/leonardotanzi/Desktop/PythonScripts/toInvert/Inverted/'

for image_path in glob.glob('/Users/leonardotanzi/Desktop/PythonScripts/toInvert/*.png'):

	image_name = image_path.split('/')[-1]

	image = Image.open(image_path)

	inverted_image = PIL.ImageOps.invert(image)

	inverted_image.save(out_path + image_name)