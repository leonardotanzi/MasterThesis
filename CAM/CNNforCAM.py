from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input as pre_process_VGG
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input as pre_process_ResNet
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input as pre_process_Inception
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras import backend as K
import argparse
import numpy as np
import time
import os
import tensorflow as tf


if __name__ == "__main__":

        ap = argparse.ArgumentParser()
        ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
        args = vars(ap.parse_args())
        run_on_server = args["server"]

        image_size = 224
        n_fold = 2
        n_class = 3
        accuracies = [[] for x in range(n_class)]
        best_accuracies = [[] for x in range(n_class)]
        scores = [[] for x in range(2)]
        best_scores = [[] for x in range(2)]

        for i in range(1, n_fold+1):
                if run_on_server == "y":
                        train_folder = "/mnt/Data/ltanzi/SUBGROUPS_A/SubgroupA_Proportioned/Fold{}/Train".format(i)
                        val_folder = "/mnt/Data/ltanzi/SUBGROUPS_A/SubgroupA_Proportioned/Fold{}/Validation".format(i)
                        test_folder = "/mnt/Data/ltanzi/SUBGROUPS_A/SubgroupA_Proportioned/Test"
                        out_folder = "/mnt/Data/ltanzi/networksForCam/"

                elif run_on_server == "n":
                        train_folder = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Proportioned/Fold{}/Train".format(i)
                        val_folder = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Proportioned/Fold{}/Validation".format(i)
                        test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Proportioned/Test".format(i)
                        out_folder = "/Users/leonardotanzi/Desktop/"

                else:
                        raise ValueError("Incorrect 1st arg")

                print("Fold number {}".format(i))

                binary = "categorical"
                loss = "sparse_categorical_crossentropy"
                last_layer = 3
                classmode = "sparse"
                act = "softmax"
                classes = ["A1", "A2", "A3"]
                name = "Fold{}_VGGforCAMA1A2A3".format(i)

                es = EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=10)  # verbose to print the n of epoch in which stopped,
                best_model_path = out_folder + name + "-best_model.h5"
                mc = ModelCheckpoint(best_model_path, monitor="val_acc", save_best_only=True, mode='max', verbose=1)

                input_shape = (224, 224, 3)

                initial_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape, pooling="avg")
                last = initial_model.output
                prediction = Dense(3, activation="softmax")(last)
                model = Model(initial_model.input, prediction)
                model.summary()

                preprocess_input = pre_process_VGG

                adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)

                model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])

                # Fit model
                data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                        preprocessing_function=preprocess_input)
                data_generator_notAug = ImageDataGenerator(preprocessing_function=preprocess_input)

                # Takes the path to a directory & generates batches of augmented data.
                train_generator = data_generator.flow_from_directory(train_folder,
                        target_size=(image_size, image_size),
                        batch_size=32,
                        class_mode=classmode,
                        classes=classes)

                validation_generator = data_generator_notAug.flow_from_directory(val_folder,
                        target_size=(image_size, image_size),
                        batch_size=32,
                        class_mode=classmode,
                        classes=classes)

                test_generator = data_generator_notAug.flow_from_directory(test_folder,
                        target_size=(image_size, image_size),
                        batch_size=32,
                        class_mode=classmode,
                        classes=classes)

                # Trains the model on data generated batch-by-batch by a Python generator
                # When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs.

                STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
                STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
                STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

                # fit_generator calls train_generator that generate a batch of images from train_folder

                model.fit_generator(
                        train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=150,
                        validation_data=validation_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[es, mc])

                # my_new_model.summary()
                # plot_model(my_new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

                model.save(out_folder + name + ".model")

                # EVALUATION
                score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
                print("EVALUATING MODEL")
                print("Test loss:", score[0])
                print("Test accuracy:", score[1])
                scores[0].append(score[0])
                scores[1].append(score[1])

                best_model = load_model(best_model_path)
                best_score = best_model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
                print("EVALUATING BEST MODEL")
                print("Test loss:", best_score[0])
                print("Test accuracy:", best_score[1])
                best_scores[0].append(best_score[0])
                best_scores[1].append(best_score[1])
