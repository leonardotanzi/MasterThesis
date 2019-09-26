from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input as pre_process_VGG
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input as pre_process_ResNet
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input as pre_process_Inception
from tensorflow.python.keras.models import Sequential, load_model
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
        ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
        ap.add_argument("-m", "--model", required=True, help="Select the network (0 for VGG, 1 for ResNet, 2 for InceptionV3)")
        args = vars(ap.parse_args())
        run_on_server = args["server"]
        run_binary = args["binary"]
        run_model = int(args["model"])

        models = ["VGG", "ResNet", "Inception"]
        model_type = models[run_model]
        image_size = 224 if run_model == 0 or run_model == 1 else 299
        n_class = 2 if run_binary == "y" else 3
        n_fold = 5
        accuracies = [[] for x in range(n_class)]
        best_accuracies = [[] for x in range(n_class)]
        scores = [[] for x in range(2)]
        best_scores = [[] for x in range(2)]

        for i in range(1, n_fold+1):

                if run_on_server == "y":
                        train_folder = "/mnt/data/ltanzi/SubgroupA_Proportioned_edged/Fold{}/Train".format(i)  #"/mnt/Data/ltanzi/flippedA1A2A3CrossVal/Fold{}/Train".format(i)
                        val_folder = "/mnt/data/ltanzi/SubgroupA_Proportioned_edged/Fold{}/Validation".format(i) #/mnt/Data/ltanzi/flippedA1A2A3CrossVal/Fold{}/Validation".format(i)
                        test_folder = "/mnt/data/ltanzi/SubgroupA_Proportioned_edged/Test" #"/mnt/Data/ltanzi/flippedA1A2A3CrossVal/Test"
                        out_folder = "/mnt/Data/ltanzi/A1A2A3/edged/"

                elif run_on_server == "n":
                        train_folder = "/Users/leonardotanzi/Desktop/SubgroupA_folds/Fold{}/Train".format(i)
                        val_folder = "/Users/leonardotanzi/Desktop/SubgroupA_folds/Fold{}/Validation".format(i)
                        test_folder = "/Users/leonardotanzi/Desktop/SubgroupA_folds/Test".format(i)
                        out_folder = "/Users/leonardotanzi/Desktop/SubgroupA_folds/"

                else:
                        raise ValueError("Incorrect 1st arg")

                print("Fold number {}".format(i))

                if run_binary == "y":
                        # binary = "binary"
                        # loss = "binary_crossentropy"
                        # last_layer = 1
                        # classmode = "binary"
                        # act = "sigmoid"
                        binary = "categorical"
                        loss = "sparse_categorical_crossentropy"
                        last_layer = 2
                        classmode = "sparse"
                        act = "softmax"
                        classes = ["A1", "A2"]
                        name = "Fold{}_{}_{}-binary-baseline{}-{}".format(i, classes[0], classes[1], model_type, int(time.time()))

                elif run_binary == "n":
                        binary = "categorical"
                        loss = "sparse_categorical_crossentropy"
                        last_layer = 3
                        classmode = "sparse"
                        act = "softmax"
                        classes = ["A1", "A2", "A3"]
                        name = "Fold{}_A1A2A3_notflipped-edged-retrainAll-{}-{}-{}".format(i, binary, model_type, int(time.time()))

                else:
                        raise ValueError("Incorrect 2nd arg")

                # CALLBACKS
                log_dir = out_folder + "logs/{}".format(name)
                tensorboard = TensorBoard(log_dir=log_dir)
                es = EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=10)  # verbose to print the n of epoch in which stopped,
                best_model_path = out_folder + name + "-best_model.h5"
                mc = ModelCheckpoint(best_model_path, monitor="val_acc", save_best_only=True, mode='max', verbose=1)

                # LOAD WEIGHTS
                weights = "imagenet"

                my_new_model = Sequential()
                if model_type == "VGG":
                        my_new_model.add(VGG16(include_top=False, input_shape=(image_size, image_size, 3), pooling="avg", weights=weights))
                        preprocess_input = pre_process_VGG

                elif model_type == "ResNet":
                        my_new_model.add(ResNet50(include_top=False, input_shape=(image_size, image_size, 3), pooling="avg", weights=weights))
                        preprocess_input = pre_process_ResNet

                elif model_type == "Inception":
                        my_new_model.add(InceptionV3(include_top=False, input_shape=(image_size, image_size, 3), pooling="avg", weights=weights))
                        preprocess_input = pre_process_Inception

                my_new_model.add(Dense(last_layer, activation=act))
                my_new_model.layers[0].trainable = True

                adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)

                my_new_model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])

                # Fit model
                data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                        preprocessing_function=preprocess_input)
                data_generator_notAug = ImageDataGenerator(preprocessing_function=preprocess_input)

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

                my_new_model.fit_generator(
                        train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=150,
                        validation_data=validation_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=[tensorboard, es, mc])

                # my_new_model.summary()
                # plot_model(my_new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

                my_new_model.save(out_folder + name + ".model")

                # EVALUATION
                score = my_new_model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)
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

                if run_binary == "n":

                        test_generator.reset()

                        if run_on_server == "y":
                                test_folder = ["/mnt/Data/ltanzi/SubgroupA_Proportioned_edged/Testing/TestA1",
                                               "/mnt/Data/ltanzi/SubgroupA_Proportioned_edged/Testing/TestA2",
                                               "/mnt/Data/ltanzi/SubgroupA_Proportioned_edged/Testing/TestA3"]
                                batch_size = 32
                        elif run_on_server == "n":
                                test_folder = ["/Users/leonardotanzi/Desktop/SubgroupA_folds/Testing/TestA1",
                                               "/Users/leonardotanzi/Desktop/SubgroupA_folds/Testing/TestA2",
                                               "/Users/leonardotanzi/Desktop/SubgroupA_folds/Testing/TestA3"]
                                batch_size = 32

                        dict_classes = {classes[0]: 0, classes[1]: 1, classes[2]: 2}

                        for k, folder in enumerate(test_folder):
                                test_generator = data_generator_notAug.flow_from_directory(folder,
                                                                        target_size=(image_size, image_size),
                                                                        batch_size=batch_size,
                                                                        class_mode=classmode)

                                STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

                                test_generator.reset()

                                pred = my_new_model.predict_generator(test_generator,
                                                          steps=STEP_SIZE_TEST,
                                                          verbose=1)

                                best_pred = best_model.predict_generator(test_generator,
                                                          steps=STEP_SIZE_TEST,
                                                          verbose=1)

                                predicted_class_indices = np.argmax(pred, axis=1)
                                best_predicted_class_indices = np.argmax(best_pred, axis=1)

                                labels = dict_classes
                                labels = dict((v, k) for k, v in labels.items())
                                predictions = [labels[k] for k in predicted_class_indices]
                                best_predictions = [labels[k] for k in best_predicted_class_indices]

                                tot = 0
                                x = 0
                                for j in predictions:
                                        tot += 1
                                        if j == classes[k]:
                                                x += 1

                                percentage = x*100/tot
                                print("Model:{} classified correctly: {}%".format(classes[k], percentage))
                                accuracies[k].append(percentage)

                                tot = 0
                                x = 0
                                for j in best_predictions:
                                        tot += 1
                                        if j == classes[k]:
                                                x += 1

                                percentage = x*100/tot
                                print("Best Model: {} classified correctly: {}%".format(classes[k], percentage))

                                best_accuracies[k].append(percentage)

                elif run_binary == "y":

                        test_generator.reset()

                        if run_on_server == "y":
                                test_folder = ["/mnt/Data/ltanzi/SubgroupA_flipped/Testing/Test{}".format(classes[0]),
                                               "/mnt/Data/ltanzi/SubgroupA_flipped/Testing/Test{}".format(classes[1])]
                                batch_size = 32
                        elif run_on_server == "n":
                                test_folder = ["/Users/leonardotanzi/Desktop/SubgroupA_folds/Testing/TestA1",
                                               "/Users/leonardotanzi/Desktop/SubgroupA_folds/Testing/TestA2",
                                               "/Users/leonardotanzi/Desktop/SubgroupA_folds/Testing/TestA3"]
                                batch_size = 32

                        dict_classes = {classes[0]: 0, classes[1]: 1}

                        for k, folder in enumerate(test_folder):
                                test_generator = data_generator_notAug.flow_from_directory(folder,
                                                                                           target_size=(
                                                                                           image_size,
                                                                                           image_size),
                                                                                           batch_size=batch_size,
                                                                                           class_mode=classmode)

                                STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

                                test_generator.reset()

                                pred = my_new_model.predict_generator(test_generator,
                                                                      steps=STEP_SIZE_TEST,
                                                                      verbose=1)

                                best_pred = best_model.predict_generator(test_generator,
                                                                         steps=STEP_SIZE_TEST,
                                                                         verbose=1)

                                predicted_class_indices = np.argmax(pred, axis=1)
                                best_predicted_class_indices = np.argmax(best_pred, axis=1)

                                labels = dict_classes
                                labels = dict((v, k) for k, v in labels.items())
                                predictions = [labels[k] for k in predicted_class_indices]
                                best_predictions = [labels[k] for k in best_predicted_class_indices]

                                tot = 0
                                x = 0
                                for j in predictions:
                                        tot += 1
                                        if j == classes[k]:
                                                x += 1

                                percentage = x * 100 / tot
                                print("Model:{} classified correctly: {}%".format(classes[k], percentage))
                                accuracies[k].append(percentage)

                                tot = 0
                                x = 0
                                for j in best_predictions:
                                        tot += 1
                                        if j == classes[k]:
                                                x += 1

                                percentage = x * 100 / tot
                                print("Best Model: {} classified correctly: {}%".format(classes[k], percentage))

                                best_accuracies[k].append(percentage)

        avg_accuracies = [0, 0, 0]
        avg_scores = [0, 0]
        for i in range(n_class):
                for j in range(n_fold):
                        avg_accuracies[i] += accuracies[i][j]
                avg_accuracies[i] /= n_fold

        for i in range(2):
                for j in range(n_fold):
                        avg_scores[i] += scores[i][j]
                avg_scores[i] /= n_fold

        print("MODEL")

        if run_binary == "n":
                print("Average:\n {} classified correctly {}%, {} classified correctly {}%, {} Classified correctly {}%.\n"
                      "Average loss {}, average accuracy {}".format(classes[0], avg_accuracies[0],
                                                                    classes[1], avg_accuracies[1],
                                                                    classes[2], avg_accuracies[2],
                                                                    avg_scores[0], avg_scores[1]))
        elif run_binary == "y":
                print("Average:\n {} classified correctly {}%, {} classified correctly {}%.\n"
                      "Average loss {}, average accuracy {}".format(classes[0], avg_accuracies[0],
                                                                      classes[1], avg_accuracies[1],
                                                                      avg_scores[0], avg_scores[1]))

        best_avg_accuracies = [0, 0, 0]
        best_avg_scores = [0, 0]
        for i in range(n_class):
                for j in range(n_fold):
                        best_avg_accuracies[i] += best_accuracies[i][j]
                best_avg_accuracies[i] /= n_fold

        for i in range(2):
                for j in range(n_fold):
                        best_avg_scores[i] += best_scores[i][j]
                best_avg_scores[i] /= n_fold

        print("BEST MODEL")

        if run_binary == "n":
                print("Average:\n {} classified correctly {}%, {} classified correctly {}%, {} Classified correctly {}%.\n"
                      "Average loss {}, average accuracy {}".format(classes[0], best_avg_accuracies[0], classes[1],
                                                                    best_avg_accuracies[1], classes[2],
                                                                    best_avg_accuracies[2], best_avg_scores[0],
                                                                    best_avg_scores[1]))
        elif run_binary == "y":
                print("Average:\n {} classified correctly {}%, {} classified correctly {}%.\n"
                      "Average loss {}, average accuracy {}".format(classes[0], best_avg_accuracies[0],
                                                                    classes[1], best_avg_accuracies[1],
                                                                    best_avg_scores[0], best_avg_scores[1]))
