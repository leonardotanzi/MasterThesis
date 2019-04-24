import tensorflow as tf
import cv2

CATEGORIES = ["Broken", "Unbroken"]


def prepare(filepath):
    IMG_SIZE = 256
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("firstModel.model")

# always made prediction on list that's why we have []
prediction = model.predict([prepare("unbroken_test3.jpg")])
print(CATEGORIES[int(prediction[0][0])])