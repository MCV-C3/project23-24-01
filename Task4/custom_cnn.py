import os

import tensorflow as tf
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

#canviar el nom cada vegada
MODEL_FNAME = f'/ghome/group01/group01/project23-24-01/Task4/weights/GPU/custom.weights.h5'

NUM_CLASSES=8
DATASET_DIR_CNN = '/ghome/mcv/datasets/C3/MIT_split'

# Data loading for the CNN model
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR_CNN, 'train'),
    target_size=(64, 64),
    batch_size=BATCH_SIZE,
    class_mode='categorical')  # Update class_mode for multi-class

validation_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR_CNN, 'test'),
    target_size=(64, 64),
    batch_size=BATCH_SIZE,
    class_mode='categorical')  # Update class_mode for multi-class

# Adjust the output layer for multi-class classification
model = Sequential()
# Input layer
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output before feeding into dense layers
model.add(Flatten())

# Dense layers with dropout for regularization
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(NUM_CLASSES, activation='softmax'))  # Update NUM_CLASSES based on your dataset

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
