from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
import os
import tensorflow as tf
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Constants
NUM_CLASSES = 3
IMAGE_SIZE = 224
IMAGE_NUM_CHANNELS = 3
TRAIN_IMAGES_PER_CLASS = 500
NUM_EPOCHS = 5
TRAIN_IMAGES_FOLDER = 'images/training_traffic_signs'
MODEL_FOLDER = 'trainedModels/model_traffic_signs'
#FILE_ENDING = '.png'
FILE_ENDING = '.ppm'
#CLASS_NAME_ARRAY = ['1 (Square)' , '2 (Circle)', '3 (Triangle)']
CLASS_NAME_ARRAY = ['1 (Right of way)' , '2 (Give way)', '3 (Stop)']
TRAIN_PATH_TO_FILES_ARRAY = [TRAIN_IMAGES_FOLDER+'/1/', TRAIN_IMAGES_FOLDER+'/2/', TRAIN_IMAGES_FOLDER+'/3/']

SHOW_IMAGES = True
TRAIN_MODE = True
PLOT_HISTORY = True
PREDICT_MODE = False

trainsets_array = []
global cnn_model
global cnn_model_loaded
global history

# Trainset class
class Trainset:
    def __init__(self, class_name, class_id, train_path_to_files):
        self.class_name = class_name
        self.class_id = class_id
        self.train_path_to_files = train_path_to_files
        self.train_file_name_set = []
        self.train_num_images = 0
        self.train_img_data = []

# Load training images
def load_train_images():
    # Create Trainset for all classes
    for c in range(0,NUM_CLASSES):
        # Create Trainset object
        trainset = Trainset(CLASS_NAME_ARRAY[c], c+1, TRAIN_PATH_TO_FILES_ARRAY[c])
        # Get all files in the current folder
        files_in_folder = os.listdir(trainset.train_path_to_files)
        # Filter out image files
        for file in files_in_folder:
            if(file.endswith(FILE_ENDING)):
                trainset.train_file_name_set.append(file)
        # Get number of images
        trainset.train_num_images = len(trainset.train_file_name_set)
        # Get the image data
        trainset.train_img_data = np.zeros((trainset.train_num_images, IMAGE_SIZE, IMAGE_SIZE, IMAGE_NUM_CHANNELS), dtype=int)
        for i in range(0, trainset.train_num_images):
            file_name = trainset.train_file_name_set[i]
            path_to_file = trainset.train_path_to_files+file_name
            img = Image.open(path_to_file)
            img_data = asarray(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
            trainset.train_img_data[i, :, :, :] = img_data[:,:,0:3]
        # Save trainset object
        trainsets_array.append(trainset)

# Show training images
def show_train_images():
    for i in range(0, NUM_CLASSES):
        # Print overview
        print('----------')
        print('class name: '+str(trainsets_array[i].class_name))
        print('class id: ' + str(trainsets_array[i].class_id))
        print('path: '+str(trainsets_array[i].train_path_to_files))
        print('num: '+str(trainsets_array[i].train_num_images))
        print('----------')
        # Show one image
        img_data = trainsets_array[i].train_img_data[0, :, :, :]
        plt.imshow(img_data)
        plt.show()

# Build CNN Model
def build_cnn_model():
    global cnn_model
    # Input Layer
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input")
    # Layer 1 - Convolutions
    l1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, padding="same")(inputs)
    l1 = tf.keras.layers.BatchNormalization()(l1)
    l1 = tf.keras.layers.ReLU()(l1)
    l1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l1)
    # Layer 2 - Convolutions
    l2 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, padding="same")(l1)
    l2 = tf.keras.layers.BatchNormalization()(l2)
    l2 = tf.keras.layers.ReLU()(l2)
    l2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l2)
    # Layer 3 - Convolutions
    l3 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same")(l2)
    l3 = tf.keras.layers.ReLU()(l3)
    # Layer 4 - Convolutions
    l4 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding="same")(l3)
    l4 = tf.keras.layers.ReLU()(l4)
    # Layer 5 - Convolutions
    l5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(l4)
    l5 = tf.keras.layers.ReLU()(l5)
    l5 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l5)
    # Layer 6 - Dense
    l6_pre = tf.keras.layers.Flatten()(l5)
    l6 = tf.keras.layers.Dense(units=4096)(l6_pre)
    l6 = tf.keras.layers.ReLU()(l6)
    l6 = tf.keras.layers.Dropout(rate=0.5)(l6)
    # Layer 7 - Dense
    l7 = tf.keras.layers.Dense(units=4096)(l6)
    l7 = tf.keras.layers.ReLU()(l7)
    l7 = tf.keras.layers.Dropout(rate=0.5)(l7)
    # Layer 8 - Dense
    l8 = tf.keras.layers.Dense(units=3)(l7)
    l8 = tf.keras.layers.Softmax(dtype=tf.float32, name="output")(l8)
    cnn_model = tf.keras.models.Model(inputs=inputs, outputs=l8)
    print("Building complete")

# Compile CNN Model
def compile_cnn_model():
    global cnn_model
    # Metrics
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
    # Compile
    cnn_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.01, momentum=0.9, nesterov=False, name='SGD'
        ),
        metrics=metrics,
    )
    # Plot image of CNN Model
    #tf.keras.utils.plot_model(cnn_model, to_file='cnn_model.png', show_layer_names=False, show_shapes=True, show_dtype=True, dpi=200)
    print("Compiling complete")

# Train CNN Model
def train_cnn_model():
    global cnn_model
    global history
    # Train images
    x_trainset_class_1 = trainsets_array[0].train_img_data[0:TRAIN_IMAGES_PER_CLASS, :, :, :]
    x_trainset_class_2 = trainsets_array[1].train_img_data[0:TRAIN_IMAGES_PER_CLASS, :, :, :]
    x_trainset_class_3 = trainsets_array[2].train_img_data[0:TRAIN_IMAGES_PER_CLASS, :, :, :]
    x_trainset = np.vstack((x_trainset_class_1,x_trainset_class_2,x_trainset_class_3))
    # Train labels
    y_trainlabels_class_1 = np.repeat(np.matrix([1, 0, 0]), TRAIN_IMAGES_PER_CLASS, axis=0)
    y_trainlabels_class_2 = np.repeat(np.matrix([0, 1, 0]), TRAIN_IMAGES_PER_CLASS, axis=0)
    y_trainlabels_class_3 = np.repeat(np.matrix([0, 0, 1]), TRAIN_IMAGES_PER_CLASS, axis=0)
    y_trainlabels = np.vstack((y_trainlabels_class_1,y_trainlabels_class_2,y_trainlabels_class_3))
    # Training
    print('Training Images:')
    print(x_trainset.shape)
    print('Training Labels:')
    print(y_trainlabels.shape)
    history = cnn_model.fit(epochs=NUM_EPOCHS , x=x_trainset , y=y_trainlabels)
    print("Training complete")

# Export CNN Model
def export_cnn_model():
    global cnn_model
    cnn_model.save(MODEL_FOLDER)

# Import CNN Model
def import_cnn_model():
    global cnn_model_loaded
    cnn_model_loaded = tf.keras.models.load_model(MODEL_FOLDER)

# Predict with CNN Model
def predict_cnn_model():
    global cnn_model_loaded
    # Class 1
    print("Class 1")
    test_images = trainsets_array[0].train_img_data[0:3, :, :, :]
    output = cnn_model_loaded.predict(x=test_images)
    print(output)
    # Class 2
    print("Class 2")
    test_images = trainsets_array[1].train_img_data[0:3, :, :, :]
    output = cnn_model_loaded.predict(x=test_images)
    print(output)
    # Class 3
    print("Class 3")
    test_images = trainsets_array[2].train_img_data[0:3, :, :, :]
    output = cnn_model_loaded.predict(x=test_images)
    print(output)

# Plot History
def plot_history():
    global history
    print(history.params)
    print(history.history.keys())
    keys = ['loss' , 'categorical_accuracy', 'false_negatives', 'false_positives', 'precision', 'recall']
    for key in keys:
        plt.plot(history.history[key])
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.grid()
        plt.savefig(MODEL_FOLDER + '/'+key+'.png')
        plt.clf()

# Main
if __name__ == '__main__':
    # Load training images
    load_train_images()
    if SHOW_IMAGES:
        # Show training images
        show_train_images()
    if TRAIN_MODE:
        # Build CNN Model
        build_cnn_model()
        cnn_model.summary()
        # Compile CNN Model
        compile_cnn_model()
        # Train CNN Model
        train_cnn_model()
        # Export CNN Model
        export_cnn_model()
        if PLOT_HISTORY:
            # Plot History
            plot_history()
    if PREDICT_MODE:
        # Import CNN Model
        import_cnn_model()
        # Predict with CNN Model
        predict_cnn_model()