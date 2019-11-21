from tensorflow.python.keras.applications.vgg16 import preprocess_input as pre_process_VGG
from tensorflow.python.keras.applications.resnet50 import preprocess_input as pre_process_ResNet
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as pre_process_Inception
from keras.preprocessing import image
import keras.backend as K
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Sequential
from keras.models import load_model

import numpy as np
import cv2
import glob
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-b", "--binary", required=True, help="Running the code on a binary dataset or not (y/n)")
ap.add_argument("-m", "--model", required=True, help="Select the network (0 for VGG, 1 for ResNet, 2 for InceptionV3)")
ap.add_argument("-c", "--classification", required=True, help="Tackling A, B, U (0) or A1, A2, A3 classification (1)")
args = vars(ap.parse_args())

run_on_server = args["server"]
run_binary = args["binary"]
run_model = int(args["model"])
run_classification = int(args["classification"])

if run_on_server == "y":
    model_path = "/mnt/data/ltanzi/retrainAll-categorical-baselineVGG-1562590231.model"
    test_folder = "/mnt/data/ltanzi/Train_Val/Test/A"
    out_folder = "/mnt/data/ltanzi/Cam_output"

elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/CAM/ResNet/Models/Fold2_ResNetforCAM_ABU.model"
    test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/Cascade/Test/B"
    out_folder = "/Users/leonardotanzi/Desktop/CAM/ResNet/Output/B/"

else:
    raise ValueError("Incorrect 1st arg.")


if run_binary == "n" and run_classification == 0:
    name_indexes = ["A", "B", "Unbroken"]
elif run_binary == "n" and run_classification == 1:
        name_indexes = ["A1", "A2", "A3"]
elif run_binary == "y" and run_classification == 0:
    name_indexes = ["A", "B"]
elif run_binary == "y" and run_classification == 1:
    name_indexes = ["A1", "A2"]

models = ["VGG", "ResNet", "Inception"]
model_type = models[run_model]
img_size = 224 if run_model == 0 or run_model == 1 else 299

if model_type == "VGG":
    preprocess_input = pre_process_VGG
    last_layer_name = "block5_conv3"

elif model_type == "ResNet":
    preprocess_input = pre_process_ResNet
    last_layer_name = "activation_98"

elif model_type == "Inception":
    preprocess_input = pre_process_Inception
    last_layer_name = "mixed10"

model = load_model(model_path)

model.summary()

for img_path in sorted(glob.glob(test_folder + "/*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # model.summary()

    preds = model.predict(x)

    if run_binary == "n":
        class_idx = np.argmax(preds[0])
    elif run_binary == "y":
        class_idx = int(round(preds[0][0]))  # per far funzionare binary devo re-train con softmax e non binary

    # model.output è ciò che esce dalla softmax per tre classi, io vado a prendere la strided slice relativa alla
    # posizione della classe predetta, nel riassunto di teoria fatto da me sarebbe Yc
    class_output = model.output[:, class_idx]

    # estraggo l'ultimo livello convoluzionale e prendo l'output, sono gli Ak
    last_conv_layer = model.get_layer(last_layer_name)
    conv_out = last_conv_layer.output

    # calcolo il gradiente tra Yc e Ak, quanto varia la score assegnata alla classe c in relazione alle varie maps
    grads = K.gradients(class_output, conv_out)[0]

    # faccio global avg pooling
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # definiscono una funzione che prende come ingresso placeholder dell'input del modello e restituisce placeholder
    # del gradient dopo il GAP e di un Ak. i placeholder è come se specificassero il tipo di formato che entra ed esce
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    # prende x in input e restituisce il gradiente dopo il GAP e gli Ak
    pooled_grads_value, conv_layer_output_value = iterate([x])

    # poi moltiplica ogni peso per gli Ak, quindi Wc(k)*Ak(i,j)
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # conv_layer_output_value è composto da 512 feature 14x14, con np.mean sull'asse dei channel ottengo un heatmap
    # 14x14, cioè faccio la media fra le 512 feature maps (già pesate)
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # applico la relu, cioè prendo solo i valori maggiori di 0
    heatmap = np.maximum(heatmap, 0)

    #normalizzo i valori fra 0 e 1
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 0.6 e 0.4 sono i pesi dati a img a heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    window_name = "{}-predicted {}".format(img_path.split("/")[-1].split(".")[0], name_indexes[class_idx])
    '''
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)
    cv2.moveWindow(window_name, 200, 0)

    numpy_horizontal = np.hstack((img, superimposed_img))

    cv2.imshow(window_name, numpy_horizontal)
    
    cv2.waitKey(0)
    '''
    cv2.imwrite(out_folder + window_name + ".png", superimposed_img)
