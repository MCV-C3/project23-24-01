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
import pickle
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")

wandb.init(
    # set the wandb project where this run will be logged
    project="c3_project",

    # track hyperparameters and run metadata with wandb.config
    config={"optimizer": "sgd",
            "loss": "categorical_crossentropy",
            "metric": "accuracy"}
)

config = wandb.config

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
MODEL_FNAME = '/ghome/group01/weights/20240111_20_27.weights.h5'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()


print('Setting up data ...\n')


# Load and preprocess the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
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
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
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
preprocessing_train = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
])

preprocessing_validation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
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
model.add(Dense(units=128, activation='relu', name='last'))
model.add(Dense(units=8, activation='softmax',name='classification'))
model.compile(loss=config.loss,
              optimizer=config.optimizer,
              metrics=[config.metric])

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

print('Start training...\n')
history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        verbose=0,
        callbacks=[
                      WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("models")
                    ])

wandb.finish()

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

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
layer = 'last'
model_layer = keras.Model(inputs=input, outputs=model.get_layer(layer).output)

print("Saving training output features...")
train_features = []
for x, _ in train_dataset:
  features = model_layer.predict(x/255.0)
  train_features.append(features)

train_features = np.vstack(train_features)

with open('training_features.dat', 'wb') as file:
    pickle.dump(train_features, file)


print("Saving test output features...")
test_features = []
for x, _ in test_dataset:
  features = model_layer.predict(x/255.0)
  test_features.append(features)

test_features = np.vstack(test_features)

with open('test_features.dat', 'wb') as file:
    pickle.dump(test_features, file)

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