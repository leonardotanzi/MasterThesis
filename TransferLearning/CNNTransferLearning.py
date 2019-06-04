from tensorflow.python.keras.applications import ResNet50, inception_v3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import argparse
import numpy as np
import time

image_size = 256

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]
run_binary = args["binary"]

if run_on_server == "y" and run_binary == "y":
        train_folder = "/mnt/Data/ltanzi/A_B/Train"
        val_folder = "/mnt/Data/ltanzi/A_B/Validation"
        test_folder = "/mnt/Data/ltanzi/A_B/Test"
        out_folder = "/mnt/Data/ltanzi/"
        loss = "binary_crossentropy"
        last_layer = 1
        classmode = "binary"
        act = "sigmoid"

elif run_on_server == "y" and run_binary == "n":
        train_folder = "/mnt/Data/ltanzi/Train_Val/Train"
        val_folder = "/mnt/Data/ltanzi/Train_Val/Validation"
        test_folder = "/mnt/Data/ltanzi/Train_Val/TestB"
        out_folder = "/mnt/Data/ltanzi/"
        loss = "sparse_categorical_crossentropy"
        num_classes = 3
        last_layer = 3
        classmode = "sparse"
        act = "softmax"
        
elif run_on_server == "n" and run_binary == "y":
        train_folder = "/Users/leonardotanzi/Desktop/FinalDataset/BinaryDataset/Bro_Unbro/Train"
        val_folder = "/Users/leonardotanzi/Desktop/FinalDataset/BinaryDataset/Bro_Unbro/Validation"
        test_folder = "/Users/leonardotanzi/Desktop/FinalDataset/BinaryDataset/Bro_Unbro/Test"
        out_folder = "/Users/leonardotanzi/Desktop/FinalDataset/"
        loss = "binary_crossentropy"
        last_layer = 1
        classmode = "binary"
        act = "sigmoid"

elif run_on_server == "n" and run_binary == "n":
        train_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Train"
        val_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Validation"
        test_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Test"
        out_folder = "/Users/leonardotanzi/Desktop/FinalDataset/"
        loss = "sparse_categorical_crossentropy"
        last_layer = 3
        classmode = "sparse"
        act = "softmax"
else:
        raise ValueError('Incorrect arg')

name = "ResNet-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(name))   
es = EarlyStopping(monitor="val_acc", mode = "max", verbose=1, patience=3) # verbose to print the n of epoch in which stopped, patience to wait still some epochs before stop
# mc = ModelCheckpoint(out_folder + "best_model.h5", monitor="val_acc", mode='max', verbose=1)

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling="avg", weights='imagenet'))
# my_new_model.add(Dense(32, activation="relu"))
# my_new_model.add(Dropout(0.25))
my_new_model.add(Dense(last_layer, activation=act))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)

my_new_model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])


# Fit model
data_generator = ImageDataGenerator(zca_whitening=True, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip=True, preprocessing_function=preprocess_input)

# Takes the path to a directory & generates batches of augmented data.
train_generator = data_generator.flow_from_directory(train_folder,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode=classmode)

validation_generator = data_generator.flow_from_directory(val_folder,
        target_size=(image_size, image_size),
        class_mode=classmode)

# test_generator = data_generator.flow_from_directory(test_folder,
        # target_size=(image_size, image_size),
        # batch_size=24,
        # class_mode=classmode)

# Trains the model on data generated batch-by-batch by a Python generator
# When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs.

STEP_SIZE_TRAIN =  train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
# STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        callbacks=[tensorboard, es])


my_new_model.summary()
# plot_model(my_new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

my_new_model.save(out_folder + "transferLearningAB.model")


'''
score = my_new_model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

test_generator.reset()

pred = my_new_model.predict_generator(test_generator,
        steps=STEP_SIZE_TEST,
        verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)

labels = train_generator.class_indices
labels = dict((v, k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

print(predictions)
'''
