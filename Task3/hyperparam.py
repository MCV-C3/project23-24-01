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

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import optuna

print(f'gpus? {keras.distribution.list_devices(device_type="GPU")}')
print('GPU name: ', tf.config.list_physical_devices('GPU'))

wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
IMG_WIDTH = 224
IMG_HEIGHT=224
BATCH_SIZE=32
NUMBER_OF_EPOCHS=20

train_dataset = None
test_dataset = None
validation_dataset = None

def get_datasets(batch_size):
    train_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
    )

    # Load and preprocess the training dataset
    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR+'/train/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )


    # Load and preprocess the validation dataset
    validation_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=DATASET_DIR+'/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the test dataset
    test_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_dataset = test_data_generator.flow_from_directory(
        directory=DATASET_DIR+'/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    return train_dataset, validation_dataset, test_dataset

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam'])
    learn_rate = trial.suggest_loguniform('learn_rate', 1e-4, 0.3)
    momentum = trial.suggest_uniform('momentum', 0.0, 0.9)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.8)
    l2 = trial.suggest_loguniform('l2', 0.0, 0.05)

    wandb.init(project="task3_hparam_opt", config=trial.params)

    train_dataset, validation_dataset, test_dataset = get_datasets(batch_size=batch_size)

    # create the base pre-trained model
    base_model = Xception(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    plot_model(model, to_file='modelXception.png', show_shapes=True, show_layer_names=True)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_dataset,
                        verbose=0)



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)

# Print the best hyperparameters and result
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')