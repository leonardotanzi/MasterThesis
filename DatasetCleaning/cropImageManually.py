# import the necessary packages
import argparse
import cv2
import glob
import os

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 10)


if __name__ == "__main__":

	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True, help="Path to the image")
	# args = vars(ap.parse_args())

	# load the image, clone it, and setup the mouse callback function
	# path = args["image"]

	path = "/Users/leonardotanzi/Desktop/OutLabelled/Unbroken/"  # "/Users/leonardotanzi/Desktop/rifare/"

	for image_path in sorted(glob.glob(path + "*.png"), key=os.path.getsize):

		image = cv2.imread(image_path)
		clone = image.copy()
		window_name = image_path.split('/')[-1].split('.')[0]
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(window_name, 900, 900)
		cv2.moveWindow(window_name, 200, 0)

		for i in range(2):
			cv2.setMouseCallback(window_name, click_and_crop)

			# keep looping until the 'q' key is pressed
			while True:
				# display the image and wait for a keypress
				cv2.imshow(window_name, image)

				key = cv2.waitKey(1) & 0xFF

				# if the 'r' key is pressed, reset the cropping region
				if key == ord("r"):
					image = clone.copy()

				# if the 'c' key is pressed, break from the loop
				elif key == ord("c"):
					break

			# if there are two reference points, then crop the region of interest
			# from the image and display it
			if len(refPt) == 2:
				roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
				# dim_out = (256, 256)
				if i == 0:
					cv2.imwrite("{}{}_left.png".format(path, window_name), roi)  # cv2.resize(roi, dim_out))
				elif i == 1:
					cv2.imwrite("{}{}_right.png".format(path, window_name), roi)  # cv2.resize(roi, dim_out))

		# close all open windows
		cv2.destroyAllWindows()
