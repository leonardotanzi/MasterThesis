import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]

class1 = "Broken"
class2 = "Unbroken"

subclass1 = "A"
subclass2 = "B"

if run_on_server == "y":
    score_folder = "/mnt/Data/ltanzi/Train_Val/Test"
    model_path = "/mnt/Data/ltanzi/"
    test_folder = ["/mnt/data/ltanzi/Train_Val/Testing/Test" + class1, "/mnt/data/ltanzi/Train_Val/Testing/Test" + class2]


elif run_on_server == "n":
    model_path = "/Users/leonardotanzi/Desktop/FinalDataset/"
    score_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Test"
    test_folder = ["/Users/leonardotanzi/Desktop/FinalDataset/Testing/Test" + class1, "/Users/leonardotanzi/Desktop/FinalDataset/Testing/Test" + class2]

else:
    raise ValueError("Incorrect arg.")


classmode = "binary"
image_size = 224
dict_classes = {class1: 0, class2: 1}
classes = [class1, class2]

data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, preprocessing_function=preprocess_input)

model = load_model(model_path + "Broken_Unbroken-binary-baselineVGG-1562672986-best_model.h5")


for img_path in sorted(glob.glob(test_folder + "/*.png"), key=os.path.getsize):

    img = image.load_img(img_path, target_size=(224, 224))

    X_original = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
    
            
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    class_idx = np.argmax(preds[0])

    if class_idx == "0":
        print("unbroken")

    elif class_idx == "1":
        name_out = output_path + "{}_{}".format(img_path.split("/")[-1])
        cv2.imwrite(name_out, X_original)


for img_path in sorted(glob.glob(output_path + "/*.png"), key=os.path.getsize):

    # second model eval
    # predictions



test_generator = data_generator.flow_from_directory(folder,
                                                    target_size=(image_size, image_size),
                                                    batch_size=24,
                                                    class_mode=classmode)

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

test_generator.reset()

pred = model.predict_generator(test_generator,
                              steps=STEP_SIZE_TEST,
                              verbose=1)

sqArray = np.squeeze(pred)
integer_predictions = []

for p in sqArray:
    integer_predictions.append(int(round(p)))

labels = dict_classes
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in integer_predictions]

print(predictions)

x = 0
for j in predictions:
    if j == classes[i]:
        x += 1

print("{} classified correctly: {}%".format(classes[i], x))
