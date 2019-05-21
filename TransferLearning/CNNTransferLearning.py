from tensorflow.keras.applications import ResNet50, inception_v3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model



num_classes = 3
image_size = 256
resnet_weights_path = "/home/ltanzi//MasterThesis/TransferLearning/" \
                      "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='max', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit model
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Takes the path to a directory & generates batches of augmented data.
train_generator = data_generator.flow_from_directory("/mnt/Data/ltanzi/Train_Val/Train",
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory("/mnt/Data/ltanzi/Train_Val/Validation",
        target_size=(image_size, image_size),
        class_mode='categorical')

# Trains the model on data generated batch-by-batch by a Python generator
# When you use fit_generator, the number of samples processed for each epoch is batch_size * steps_per_epochs.
my_new_model.fit_generator(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=1)


my_new_model.summary()
plot_model(my_new_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

my_new_model.save("transferLearning.model")
