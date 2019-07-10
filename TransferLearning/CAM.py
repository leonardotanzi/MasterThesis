from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import keras.backend as K
from keras.models import load_model

import numpy as np
import cv2
import glob
import os
import tensorflow as tf
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

if run_on_server == "y":
    model_path = "/mnt/data/ltanzi/retrainAll-categorical-baselineVGG-1562590231.model"
    test_folder = "/mnt/data/ltanzi/Train_Val/Test/A"
    out_folder = "/mnt/data/ltanzi/Cam_output"

elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/FinalDataset/NotEagertransferLearningVGG.model"
    test_folder = "/Users/leonardotanzi/Desktop/FinalDataset/B"
    out_folder = "/Users/leonardotanzi/Desktop/FinalDataset/CAM/B/"

else:
    raise ValueError("Incorrect 1st arg.")

model = load_model(model_path)
name_indexes = ["A", "B", "Unbroken"]

for img_path in sorted(glob.glob(test_folder + "/*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])

    class_output = model.output[:, class_idx]

    extracted_vgg_model = model.layers[0]
    last_conv_layer = extracted_vgg_model.get_layer("block5_conv3")

    conv_out = [l for l in model.layers[0].layers if l.name == "block5_conv3"][0].output

    grads = K.gradients(last_conv_layer.output, conv_out)[0]

    a = model.layers[0].layers[0].input
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        pooled_grads_value, conv_layer_output_value = \
            sess.run([pooled_grads, conv_out], feed_dict={a: x})

    for i in range(512):
        conv_layer_output_value[:, :, :, i] *= pooled_grads_value[i]

    heatmap = np.squeeze(np.mean(conv_layer_output_value, axis=-1))
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    '''
    window_name = "{}-predicted:{}".format(img_path.split("/")[-1].split(".")[0], name_indexes[class_idx])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)
    cv2.moveWindow(window_name, 200, 0)

    numpy_horizontal = np.hstack((img, superimposed_img))

    cv2.imshow(window_name, numpy_horizontal)

    cv2.waitKey(0)
    '''
    window_name = "{}-predicted:{}".format(img_path.split("/")[-1].split(".")[0], name_indexes[class_idx])
    cv2.imwrite(out_folder + window_name + ".png", superimposed_img)
