import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from task3_utils import *
import tensorflow as tf
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
IMG_WIDTH = 224
IMG_HEIGHT=224
BATCH_SIZE=32
NUMBER_OF_EPOCHS=100
NUM_CLASSES=8
MODEL_FNAME = f'/ghome/group01/group01/project23-24-01/Task3/weights/GPU/{NUMBER_OF_EPOCHS}_xception.weights.h5'
MODEL_FNAME_FROZEN = f'/ghome/group01/group01/project23-24-01/Task3/weights/GPU/{NUMBER_OF_EPOCHS}_xception_frozen.weights.h5'

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import tensorflow as tf

# Define the data generator
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

# Load and preprocess the training dataset
train_dataset = train_data_generator.flow_from_directory(
    directory=DATASET_DIR+'/train/',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Load and preprocess the validation dataset
validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

validation_dataset = validation_data_generator.flow_from_directory(
    directory=DATASET_DIR+'/test/',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Load and preprocess the test dataset
test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_dataset = test_data_generator.flow_from_directory(
    directory=DATASET_DIR+'/test/',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# create the base pre-trained model
base_model = Xception(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional Xception layers
for layer in base_model.layers:
    layer.trainable = False

plot_model(model, to_file='modelXception.png', show_shapes=True, show_layer_names=True)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

# train the model on the new data for a few epochs
history = model.fit(train_dataset,
                    epochs=NUMBER_OF_EPOCHS,
                    validation_data=validation_dataset,
                    verbose=2,
                    callbacks=[EarlyStopping(monitor='loss', patience=15)])

model.save_weights(MODEL_FNAME_FROZEN)  # always save your weights after training or during training

if True:
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'results/GPU/{NUMBER_OF_EPOCHS}_accuracy_train1_frozen.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'results/GPU/{NUMBER_OF_EPOCHS}_loss_train1_frozen.jpg')
    plt.close()
    
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top N layers, i.e. we will freeze
# the first layers and unfreeze the N rest:
N=12
for layer in model.layers[:N]:
   layer.trainable = False
for layer in model.layers[N:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit(train_dataset,
                    epochs=NUMBER_OF_EPOCHS,
                    validation_data=validation_dataset,
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='loss', patience=15)])

model.save_weights(MODEL_FNAME)  # always save your weights after training or during training

result = model.evaluate(test_dataset)
print( result)
print(history.history.keys())


# list all data in history

if True:
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'results/GPU/{NUMBER_OF_EPOCHS}_accuracy_train1.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'results/GPU/{NUMBER_OF_EPOCHS}_loss_train1.jpg')
    plt.close()