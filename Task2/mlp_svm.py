import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
from datetime import datetime

from utils import *
import keras
import tensorflow as tf

# Load the saved model

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


#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'

#Build the Multi Layer Perceptron model
model = Sequential()
input = Input(shape=(IMG_SIZE, IMG_SIZE, 3,),name='input')
model.add(input) # Input tensor
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),name='reshape'))
model.add(Dense(units=2048, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='first'))
model.add(Dropout(0.7))
model.add(Dense(units=128, activation='relu', name='last'))
model.add(Dense(units=8, activation='softmax',name='classification'))
model.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

print(model.summary())

# Load the best model
model.load_weights("/ghome/group01/group01/project23-24-01/Task2/weights/mlp_svm_20240113_17_00_weights.h5")

#to get the output of a given layer
#crop the model up to a certain layer
model_layer = keras.Model(inputs=input, outputs=model.get_layer('last').output)
model_initial_layer = keras.Model(inputs=input, outputs=model.get_layer("first").output)

# get train and test labels
train_labels = pickle.load(open('data/train_labels.dat','rb')) 
test_labels = pickle.load(open('data/test_labels.dat','rb'))

print('Getting Training Features...')
# get training features
train_features = []
itrain_features = []
train_directory = DATASET_DIR+'/train'
class_folders = os.listdir(train_directory)

for class_folder in class_folders:
  class_path = os.path.join(train_directory, class_folder)
  training_image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

  for image_file in training_image_files:
    x = np.asarray(Image.open(os.path.join(class_path, image_file)))
    x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
    features = model_layer.predict(x/255.0)
    train_features.append(features)
    ifeatures = model_initial_layer.predict(x/255.0)
    itrain_features.append(ifeatures)

train_features = np.vstack(train_features)
itrain_features = np.vstack(itrain_features)
  

print('Getting Test Features...')
# get test features
test_features = []
itest_features = []
test_directory = DATASET_DIR+'/test'
class_folders = os.listdir(test_directory)

for class_folder in class_folders:
  class_path = os.path.join(test_directory, class_folder)
  test_image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

  for image_file in test_image_files:
    x = np.asarray(Image.open(os.path.join(class_path, image_file)))
    x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
    features = model_layer.predict(x/255.0)
    test_features.append(features)
    ifeatures = model_initial_layer.predict(x/255.0)
    itest_features.append(ifeatures)

test_features = np.vstack(test_features)
itest_features = np.vstack(itest_features)

# Scaling the features
scaler = StandardScaler()
scaler.fit(train_features)
strain_features = scaler.transform(train_features)
stest_features = scaler.transform(test_features)

iscaler = StandardScaler()
iscaler.fit(itrain_features)
sitrain_features = iscaler.transform(itrain_features)
sitest_features = iscaler.transform(itest_features)


# SVM Last layer
classifier = SVC(C=0.01, kernel='linear', gamma=1)
classifier.fit(train_features,train_labels)
accuracy = classifier.score(test_features, test_labels)
print('SVM Last Layer accuracy: ', accuracy)

# SVM First layer
iclassifier = SVC(C=0.01, kernel='linear', gamma=1)
iclassifier.fit(itrain_features,train_labels)
accuracy = iclassifier.score(itest_features, test_labels)
print('SVM First Layer accuracy: ', accuracy)

# Standardized last layer
sclassifier = SVC(C=0.01, kernel='linear', gamma=1)
sclassifier.fit(strain_features,train_labels)
accuracy = sclassifier.score(stest_features, test_labels)
print('SVM Standardized Last Layer accuracy: ', accuracy)

# Standardized first layer
siclassifier = SVC(C=0.01, kernel='linear', gamma=1)
siclassifier.fit(sitrain_features,train_labels)
accuracy = siclassifier.score(sitest_features, test_labels)
print('SVM Standardized First Layer accuracy: ', accuracy)

print('Done!')


