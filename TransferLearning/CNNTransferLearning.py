from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Activation
# from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import argparse
import numpy as np
import time
import os
import tensorflow as tf


def compute_weights(input_folder):
        dictio = {"A": 0, "B": 1, "Unbroken": 2}
        files_per_class = []
        for folder in os.listdir(input_folder):
            if folder.startswith('.'):
                    continue
            if not os.path.isfile(folder):
                    a=dictio.get(folder)
                    files_per_class.insert(dictio.get(folder), (len(os.listdir(input_folder + '/' + folder))))
        total_files = sum(files_per_class)
        class_weights = {}
        for i in range(len(files_per_class)):
            class_weights[i] = 1 - (float(files_per_class[i]) / total_files)
        return class_weights


def VGG16_dropout_batchnorm():
        input_shape = (224, 224, 3)

        model = Sequential([
                Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
                       activation='relu'),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Dropout(0.5),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                Conv2D(128, (3, 3), activation='relu', padding='same', ),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Dropout(0.5),
                Conv2D(256, (3, 3), activation='relu', padding='same', ),
                Conv2D(256, (3, 3), activation='relu', padding='same', ),
                Conv2D(256, (3, 3), activation='relu', padding='same', ),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Dropout(0.5),
                Conv2D(512, (3, 3), activation='relu', padding='same', ),
                Conv2D(512, (3, 3), activation='relu', padding='same', ),
                Conv2D(512, (3, 3), activation='relu', padding='same', ),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Dropout(0.5),
                Conv2D(512, (3, 3), activation='relu', padding='same', ),
                Conv2D(512, (3, 3), activation='relu', padding='same', ),
                Conv2D(512, (3, 3), activation='relu', padding='same', ),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                Dropout(0.5),
                Flatten(),
                Dense(4096),
                Activation('relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(4096),
                Activation('relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(3, activation='softmax')
        ])
        return model


if __name__ == "__main__":

        ap = argparse.ArgumentParser()
        ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
        ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
        args = vars(ap.parse_args())
        run_on_server = args["server"]
        run_binary = args["binary"]

        model_type = "VGG"
        image_size = 224

        if run_on_server == "y":
                train_folder = "/mnt/Data/ltanzi/Train_Val/Train"
                val_folder = "/mnt/Data/ltanzi/Train_Val/Validation"
                test_folder = "/mnt/Data/ltanzi/Train_Val/Test"
                out_folder = "/mnt/Data/ltanzi/"

        elif run_on_server == "n":
                train_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Train"
                val_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Validation"
                test_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Test"
                out_folder = "/Users/leonardotanzi/Desktop/FinalDataset/"

        else:
                raise ValueError("Incorrect 1st arg")

        if run_binary == "y":
                binary = "binary"
                loss = "binary_crossentropy"
                last_layer = 1
                classmode = "binary"
                act = "sigmoid"
                classes = ["B", "Unbroken"]
                name = "{}_{}-{}-baseline{}-{}".format(classes[0], classes[1], binary, model_type, int(time.time()))

        elif run_binary == "n":
                binary = "categorical"
                loss = "sparse_categorical_crossentropy"
                last_layer = 3
                classmode = "sparse"
                act = "softmax"
                classes = None
                name = "addlayers-unbalanced-{}-baseline{}-{}".format(binary, model_type, int(time.time()))

        else:
                raise ValueError("Incorrect 2nd arg")

        # class_weights_train = compute_weights(train_folder)
        tensorboard = TensorBoard(log_dir="logs/{}".format(name))
        es = EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=10)  # verbose to print the n of epoch in which stopped,
                                                                                # patience to wait still some epochs before stop

        mc = ModelCheckpoint(out_folder + name + "-best_model.h5", monitor="val_acc", save_best_only=True, mode='max', verbose=1)

        baseline = True

        if baseline:
                my_new_model = Sequential()
                # my_new_model.add(ResNet50(include_top=False, pooling="avg", weights='imagenet'))
                my_new_model.add(VGG16(include_top=False, input_shape=(image_size, image_size, 3), pooling="avg", weights="imagenet"))
                # Say not to train first layer (ResNet) model. It is already trained
                my_new_model.add(Dense(4096, activation="relu"))
                my_new_model.add(BatchNormalization())
                my_new_model.add(Dropout(0.5))
                my_new_model.add(Dense(4096, activation="relu"))
                my_new_model.add(BatchNormalization())
                my_new_model.add(Dropout(0.5))
                my_new_model.add(Dense(last_layer, activation=act))
                my_new_model.layers[0].trainable = False
        else:
                my_new_model = VGG16_dropout_batchnorm()

        adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)

        my_new_model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])

        # Fit model
        data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                horizontal_flip=True, preprocessing_function=preprocess_input)

        '''
        Keras works with batches of images. So, the first dimension is used for the number of samples (or images) you have.
        When you load a single image, you get the shape of one image, which is (size1,size2,channels).
        In order to create a batch of images, you need an additional dimension: (samples, size1,size2,channels)
        The preprocess_input function is meant to adequate your image to the format the model requires.
        Some models use images with values ranging from 0 to 1. Others from -1 to +1. Others use the "caffe" style, that is not 
        normalized, but is centered.
        From the source code, Resnet is using the caffe style.
        You don't need to worry about the internal details of preprocess_input. But ideally, you should load images with the
         keras functions for that (so you guarantee that the images you load are compatible with preprocess_input).
        
        
        First, if we are working with images, loading the entire data-set in a single python variable isn’t an option, and so we 
        need a generator function.
        A generator function is like a normal python function, but it behaves like an iterator. It has a special keyword yield, 
        which is similar to return as it returns some value. When the generator is called, it will return some value and save the
        state. Next time when we call the generator again, it will resume from the saved state, and return the next set of 
        values just like an iterator
        Thus using the advantage of generator, we can iterate over each (or batches of) image(s) in the large data-set and train
        our neural net quite easily.
        
        '''

        # Takes the path to a directory & generates batches of augmented data.
        train_generator = data_generator.flow_from_directory(train_folder,
                target_size=(image_size, image_size),
                batch_size=24,
                class_mode=classmode,
                classes=classes)

        validation_generator = data_generator.flow_from_directory(val_folder,
                target_size=(image_size, image_size),
                batch_size=24,
                class_mode=classmode,
                classes=classes)

        test_generator = data_generator.flow_from_directory(test_folder,
                target_size=(image_size, image_size),
                batch_size=24,
                class_mode=classmode,
                classes=classes)

        # Trains the model on data generated batch-by-batch by a Python generator
        # When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs.

        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
        STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

        # fit_generator calls train_generator that generate a batch of images from train_folder

        my_new_model.fit_generator(
                train_generator,
                steps_per_epoch=STEP_SIZE_TRAIN,
                epochs=100,
                validation_data=validation_generator,
                validation_steps=STEP_SIZE_VALID,
                # class_weight=class_weights_train,
                callbacks=[tensorboard, es, mc])

        my_new_model.summary()
        # plot_model(my_new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        my_new_model.save(out_folder + name + ".model")


        # EVALUATION

        score = my_new_model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        test_generator.reset()

        test_folder = ["/mnt/Data/ltanzi/Train_Val/Testing/TestA", "/mnt/Data/ltanzi/Train_Val/Testing/TestB",
               "/mnt/Data/ltanzi/Train_Val/Testing/TestUnbroken"]
        dict_classes = {'Unbroken': 2, 'B': 1, 'A': 0}
        classes = ["A", "B", "Unbroken"]

        for i, folder in enumerate(test_folder):
                test_generator = data_generator.flow_from_directory(folder,
                                                                    target_size=(image_size, image_size),
                                                                    batch_size=24,
                                                                    class_mode=classmode)

                STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

                test_generator.reset()

                pred = my_new_model.predict_generator(test_generator,
                                steps=STEP_SIZE_TEST,
                                verbose=1)

                predicted_class_indices = np.argmax(pred, axis=1)

                labels = dict_classes
                labels = dict((v, k) for k, v in labels.items())
                predictions = [labels[k] for k in predicted_class_indices]

                # print(predictions)

                x = 0
                for j in predictions:
                        if j == classes[i]:
                                x += 1

                print("{} classified correctly: {}%".format(classes[i], x))
