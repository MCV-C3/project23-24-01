import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from utils import *
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import SGD

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
IMG_WIDTH = 224
IMG_HEIGHT=224
BATCH_SIZE=32
NUMBER_OF_EPOCHS=20

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f'gpus? {keras.distribution.list_devices(device_type="GPU")}')
print('GPU name: ', tf.config.list_physical_devices('GPU'))

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