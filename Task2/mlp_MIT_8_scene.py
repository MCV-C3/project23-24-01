import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
MODEL_FNAME = '/ghome/group10/work/C3/my_first_mlp.weights.h5'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()


print('Setting up data ...\n')


# Load and preprocess the training dataset
train_dataset = keras.utils.image_dataset_from_directory(
  directory=DATASET_DIR+'/train/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(IMG_SIZE, IMG_SIZE),
  shuffle=True,
  validation_split=None,
  subset=None
)

# Load and preprocess the validation dataset
validation_dataset = keras.utils.image_dataset_from_directory(
  directory=DATASET_DIR+'/test/',
  labels='inferred',
  label_mode='categorical',
  batch_size=BATCH_SIZE,
  class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
  image_size=(IMG_SIZE, IMG_SIZE),
  shuffle=True,
  seed=123,
  validation_split=None,
  subset=None
)

# Data augmentation and preprocessing
preprocessing_train = keras.Sequential([
  keras.layers.Rescaling(1./255),
  keras.layers.RandomFlip("horizontal")
])

preprocessing_validation = keras.Sequential([
  keras.layers.Rescaling(1./255)
])

train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


print('Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
input = Input(shape=(IMG_SIZE, IMG_SIZE, 3,),name='input')
model.add(input) # Input tensor
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),name='reshape'))
model.add(Dense(units=2048, activation='relu',name='first'))
#model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=8, activation='softmax',name='classification'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

print('Start training...\n')
history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        verbose=0)


print('Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(MODEL_FNAME)  # always save your weights after training or during training

  # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.jpg')

#to get the output of a given layer
 #crop the model up to a certain layer
layer = 'first'
model_layer = keras.Model(inputs=input, outputs=model.get_layer(layer).output)

#get the features from images
directory = DATASET_DIR+'/test/coast'
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
print(f'prediction for image {os.path.join(directory, os.listdir(directory)[0] )} on  layer {layer}')
features = model_layer.predict(x/255.0)
print(features.shape)
print(features)

#get classification
classification = model.predict(x/255.0)
print(f'classification for image {os.path.join(directory, os.listdir(directory)[0] )}:')
print(classification/np.sum(classification,axis=1))

print('Done!')