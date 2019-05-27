from tensorflow.python.keras.applications import ResNet50, inception_v3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import Adam
import argparse


image_size = 256

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server", required=True, help="Running the code on the server or not (y/n)")
ap.add_argument("-b", "--binary", required=True, help="NN works on binary classification or not (y/n)")
args = vars(ap.parse_args())
run_on_server = args["server"]
run_binary = args["binary"]

if run_on_server == "y" and run_binary == "y":
        train_folder = "/mnt/Data/ltanzi/Train_Val_BROUNBRO/Train"
        val_folder = "/mnt/Data/ltanzi/Train_Val_BROUNBRO/Validation"
        out_folder = "/mnt/Data/ltanzi/"
        resnet_weights_path = "imagenet"
        loss = "binary_crossentropy"
        num_classes = 2
        last_layer = 1
        classmode = "binary"

elif run_on_server == "y" and run_binary == "n":
        train_folder = "/mnt/Data/ltanzi/Train_Val/Train"
        val_folder = "/mnt/Data/ltanzi/Train_Val/Validation"
        out_folder = "/mnt/Data/ltanzi/"
        resnet_weights_path = "imagenet"
        loss = "sparse_categorical_crossentropy"
        num_classes = 3
        last_layer = 3
        classmode = "sparse"
        
elif run_on_server == "n" and run_binary == "y":
        train_folder = "/Users/leonardotanzi/Desktop/FinalDataset/BinaryDataset/Train"
        val_folder = "/Users/leonardotanzi/Desktop/FinalDataset/BinaryDataset/Validation"
        out_folder = "/Users/leonardotanzi/Desktop/FinalDataset/"
        resnet_weights_path = "/Users/leonardotanzi/Desktop/MasterThesis/TransferLearning/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        loss = "binary_crossentropy"
        last_layer = 1
        classmode = "binary"

elif run_on_server == "n" and run_binary == "n":
        train_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Train"
        val_folder = "/Users/leonardotanzi/Desktop/FinalDataset/Train_Val/Validation"
        out_folder = "/Users/leonardotanzi/Desktop/FinalDataset/"
        resnet_weights_path = "/Users/leonardotanzi/Desktop/MasterThesis/TransferLearning/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        loss = "sparse_categorical_crossentropy"
        last_layer = 3
        classmode = "sparse"
else:
        raise ValueError('Incorrect arg')


my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(last_layer, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)

my_new_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])


# Fit model
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Takes the path to a directory & generates batches of augmented data.
train_generator = data_generator.flow_from_directory(train_folder,
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode=classmode)

validation_generator = data_generator.flow_from_directory(val_folder,
        target_size=(image_size, image_size),
        class_mode=classmode)

# Trains the model on data generated batch-by-batch by a Python generator
# When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs.

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=1)


my_new_model.summary()
# plot_model(my_new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

my_new_model.save(out_folder + "transferLearning.model")

