from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Sequential
from keras.models import load_model
from tensorflow.python.keras.layers import Dense
from keras.models import Model


import numpy as np
import cv2
import glob
import os
import time

test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Test/Unbroken"

model = VGG16(weights="imagenet")

for img_path in sorted(glob.glob(test_folder + "/*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model.summary()

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]  # extract a slice from the output referring to the index
    print(model.output)
    print(class_output)

    last_conv_layer = model.get_layer("block5_conv3")
    conv_out = last_conv_layer.output
    print(last_conv_layer)
    print(last_conv_layer.output)

    grads = K.gradients(class_output, conv_out)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    window_name = "Original-CAM"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)
    cv2.moveWindow(window_name, 200, 0)

    numpy_horizontal = np.hstack((img, superimposed_img))

    cv2.imshow(window_name, numpy_horizontal)

    cv2.waitKey(0)
