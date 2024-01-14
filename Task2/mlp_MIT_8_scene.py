import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
from datetime import datetime

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Reshape, Input, Dropout, Add
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import regularizers
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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
    project="c3_project_2",

    # track hyperparameters and run metadata with wandb.config
    config={"optimizer": "sgd",
            "loss": "categorical_crossentropy",
            "metric": "accuracy"}
)

config = wandb.config

current_datetime = datetime.now()

# Format the date and time as a string (YYYYMMDD_HH_MM)
formatted_datetime = current_datetime.strftime("%Y%m%d_%H_%M")

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
WEIGHTS_FNAME = f'//ghome/group01/group01/project23-24-01/Task2/weights/mlp_svm_{formatted_datetime}_weights.h5'
MODEL_FNAME = f'/ghome/group01/group01/project23-24-01/Task2/weights/mlp_svm_{formatted_datetime}_model.h5'
RESULTS_DIR = '/ghome/group01/group01/project23-24-01/Task2/results/mlp_svm/'


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
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3,),name='input')
model.add(input_layer) # Input tensor
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),name='reshape'))
model.add(Dense(units=2048, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='first'))
model.add(Dropout(0.7))
model.add(Dense(units=128, activation='relu', name='last'))
model.add(Dense(units=8, activation='softmax',name='classification'))
model.compile(loss=config.loss,
              optimizer=config.optimizer,
              metrics=[config.metric])

print(model.summary())
plot_model(model, to_file=f'{RESULTS_DIR}/{formatted_datetime}_modelMLP.png', show_shapes=True, show_layer_names=True)

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

# Learning Rate Schedule
def lr_schedule(epoch):
  base_lr = 0.01
  decay_rate = 0.9
  min_lr = 0.001

  if epoch > 30:
    return max(base_lr * (decay_rate ** (epoch // 25)), min_lr)
  else:
    return base_lr
  
def lr_ct(epoch):
    base_lr = 0.01
    return base_lr

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

print('Start training...\n')
history = model.fit(
        train_dataset,
        epochs=250,
        validation_data=validation_dataset,
        verbose=0,
        callbacks=[
                      WandbMetricsLogger(log_freq=5),
                      WandbModelCheckpoint("val_loss"),
                      LearningRateScheduler(lr_schedule),
                      early_stopping
                    ])

print('Saving the model into W&B \n')
model.save(os.path.join(wandb.run.dir, "model.h5"))

wandb.finish()

print('Saving the model into '+MODEL_FNAME+' \n')
model.save(MODEL_FNAME)  
model.save_weights(WEIGHTS_FNAME)
print('Done!\n')

  # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{RESULTS_DIR}/{formatted_datetime}_accuracy.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{RESULTS_DIR}/{formatted_datetime}_loss.jpg')

