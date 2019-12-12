from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input as pre_process_VGG
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input as pre_process_ResNet
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input as pre_process_Inception
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import argparse
import scipy.stats
from sklearn.utils import class_weight
import numpy as np


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


if __name__ == "__main__":

    # ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
    ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
    ap.add_argument("-m", "--model", required=True,
                    help="Select the network (0 for VGG, 1 for ResNet, 2 for InceptionV3)")
    ap.add_argument("-c", "--classification", required=True,
                    help="Tackling A, B, U (0) or A1, A2, A3 classification (1)")
    args = vars(ap.parse_args())
    run_on_server = args["server"]
    run_binary = args["binary"]
    run_model = int(args["model"])
    run_classification = int(args["classification"])

    # HYPERPARAMETERS
    models = ["VGG", "ResNet", "Inception"]
    model_type = models[run_model]
    image_size = 224 if run_model == 0 or run_model == 1 else 299
    n_fold = 5
    binary = "categorical"
    loss = "sparse_categorical_crossentropy"
    classmode = "sparse"
    act = "softmax"
    n_epochs = 150
    patience_es = 10
    batch_size = 32
    learning_rate = 0.0001

    # TESTING
    n_class = 3 if run_binary == "n" else 2
    scores = [[] for x in range(2)]
    best_scores = [[] for x in range(2)]

    for i in range(1, n_fold+1):

        print("Fold number {}".format(i))

        if run_on_server == "y":
            train_folder = "/mnt/Data/ltanzi/PAPER/All_Cross_Val/Fold{}/Train".format(i)
            val_folder = "/mnt/Data/ltanzi/PAPER/All_Cross_Val/Fold{}/Validation".format(i)
            test_folder = "/mnt/Data/ltanzi/PAPER/All_Cross_Val/Test"
            out_folder = "/mnt/data/ltanzi/PAPER/Output/Cascade/BroUnbro/Models/".format(model_type)

        elif run_on_server == "n":
            train_folder = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Folds_Proportioned/Fold{}/Train".format(i)
            val_folder = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Folds_Proportioned/Fold{}/Validation".format(i)
            test_folder = "/Users/leonardotanzi/Desktop/NeededDataset/SubgroupA_Folds_Proportioned/Test"
            out_folder = "/Users/leonardotanzi/Desktop/"
        else:
            raise ValueError("Incorrect 1st arg")

        if run_binary == "n":
            classes = ["A1", "A2", "A3"] if run_classification == 1 else ["A1", "A2", "A3", "B", "Unbroken"]
            name = "Fold{}_{}_{}{}{}".format(i, model_type, classes[0], classes[1], classes[2])
            final_model_name = "{}_{}{}{}".format(model_type, classes[0], classes[1], classes[2])
            last_layer = 5
        elif run_binary == "y":
            classes = ["A1", "A2"] if run_classification == 1 else ["Broken", "Unbroken"]
            name = "Fold{}_{}_{}{}".format(i, model_type, classes[0], classes[1])
            final_model_name = "{}_{}{}".format(model_type, classes[0], classes[1])
            last_layer = 2

        # CALLBACKS
        log_dir = out_folder + "logs/{}".format(name)
        tb = TensorBoard(log_dir=log_dir,  write_graph=True, write_grads=True, write_images=True)
        es = EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=patience_es)  # verbose to print the n of epoch in which stopped,
        best_model_path = out_folder + name + "-best_model.h5"
        mc = ModelCheckpoint(best_model_path, monitor="val_acc", save_best_only=True, mode='max', verbose=1)

        input_shape = (image_size, image_size, 3)

        if model_type == "VGG":
            initial_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape,
                                  pooling="avg")
            preprocess_input = pre_process_VGG
        elif model_type == "ResNet":
            initial_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape,
                                     pooling="avg")
            preprocess_input = pre_process_ResNet
        elif model_type == "Inception":
            initial_model = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape,
                                        pooling="avg")
            preprocess_input = pre_process_Inception

        last = initial_model.output
        prediction = Dense(last_layer, activation=act)(last)
        model = Model(initial_model.input, prediction)
        # model.summary()

        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0)

        model.compile(optimizer=adam, loss=loss, metrics=["accuracy"])

        # Fit model
        data_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                            preprocessing_function=preprocess_input)
        data_generator_notAug = ImageDataGenerator(preprocessing_function=preprocess_input)

        # Takes the path to a directory & generates batches of augmented data.
        train_generator = data_generator.flow_from_directory(train_folder,
                                                             target_size=(image_size, image_size),
                                                             batch_size=batch_size,
                                                             class_mode=classmode,
                                                             classes=classes)

        validation_generator = data_generator_notAug.flow_from_directory(val_folder,
                                                                         target_size=(image_size, image_size),
                                                                         batch_size=batch_size,
                                                                         class_mode=classmode,
                                                                         classes=classes)

        '''
        test_generator = data_generator_notAug.flow_from_directory(test_folder,
                                                                   target_size=(image_size, image_size),
                                                                   batch_size=batch_size,
                                                                   class_mode=classmode,
                                                                   classes=classes)
        '''
        # Trains the model on data generated batch-by-batch by a Python generator
        # When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs.

        STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size
        # STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

        # fit_generator calls train_generator that generate a batch of images from train_folder

        class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(train_generator.classes),
            train_generator.classes)

        model.fit_generator(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            epochs=n_epochs,
            validation_data=validation_generator,
            validation_steps=STEP_SIZE_VALID,
            class_weight=class_weights,
            callbacks=[tb, es, mc])

        model.save(out_folder + name + ".model")

        # EVALUATION
        '''
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

    loss_m, CIlos_m_low, CIlos_m_high = mean_confidence_interval(scores[0], confidence=0.95)
    loss_bm, CIlos_bm_low, CIlos_bm_high = mean_confidence_interval(best_scores[0], confidence=0.95)

    acc_m, CIacc_m_low, CIacc_m_high = mean_confidence_interval(scores[1], confidence=0.95)
    acc_bm, CIacc_bm_low, CIacc_bm_high = mean_confidence_interval(best_scores[1], confidence=0.95)

    CI_out_m = "MODEL: average loss {:0.2f} (CI {:0.2f}-{:0.2f}) average accuracy {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(
        loss_m, CIlos_m_low, CIlos_m_high, acc_m, CIacc_m_low, CIacc_m_high)
    CI_out_bm = "BEST MODEL: average loss {:0.2f} (CI {:0.2f}-{:0.2f}) average accuracy {:0.2f} (CI {:0.2f}-{:0.2f})\n".format(
        loss_bm, CIlos_bm_low, CIlos_bm_high, acc_bm, CIacc_bm_low, CIacc_bm_high)

    print(CI_out_m)
    print(CI_out_bm)

    
    avg_sc_acc = np.mean(scores[1])
    std_sc_acc = np.std(scores[1])
    avg_sc_los = np.mean(scores[0])
    std_sc_los = np.std(scores[0])

    avg_bsc_acc = np.mean(best_scores[1])
    std_bsc_acc = np.std(best_scores[1])
    avg_bsc_los = np.mean(best_scores[0])
    std_bsc_los = np.std(best_scores[0])

    model_out_text = "MODEL: average loss {:0.4f} ({:0.4f})% average accuracy {:0.4f} ({:0.4f})%\n".format(
        avg_sc_los, std_sc_los, avg_sc_acc, std_sc_acc)
    best_model_out_text = "BEST MODEL: average loss {:0.4f} ({:0.4f})% average accuracy {:0.4f} ({:0.4f})%\n".format(
        avg_bsc_los, std_bsc_los, avg_bsc_acc, std_bsc_acc)

    print(model_out_text)
    print(best_model_out_text)

    file_path = "/Users/leonardotanzi/Desktop/{}summary.txt".format(final_model_name)
    file = open(file_path, "a")
    file.write("Run the model {}\nEpochs: {}\nPatience: {}\nBatch size: {}\nLearning Rate: {}\n".format(
        final_model_name, n_epochs, patience_es, batch_size, learning_rate))
    file.write("RESULTS\n")
    file.write(model_out_text)
    file.write(best_model_out_text)
    file.write(CI_out_m)
    file.write(CI_out_bm)
    file.close()
    '''
