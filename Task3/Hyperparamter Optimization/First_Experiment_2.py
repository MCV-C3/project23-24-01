import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.utils import plot_model
from keras.optimizers import Adadelta, Adagrad, Adam

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import regularizers

import optuna

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'
IMG_WIDTH = 224
IMG_HEIGHT=224
NUMBER_OF_EPOCHS=30


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
        directory=DATASET_DIR_GLOBAL+'/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )


    return train_dataset, validation_dataset, test_dataset

wandb.finish()
train_dataset, validation_dataset, test_dataset = get_datasets(batch_size=16)

def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer',  ['Adagrad', 'Adadelta', 'Adam']) #, 'adagrad', 'adadelta', 'adam'
    learn_rate = trial.suggest_categorical('learn_rate', [0.01, 0.02, 0.05, 0.1])
    momentum = trial.suggest_float('momentum', 0.0, 0.9)
    lmbda = trial.suggest_categorical('l2', [0.0, 0.001, 0.005, 0.01])
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)

    wandb.init(project="task3_hparam_opt_ada", config=trial.params, name=f"run_{trial.number}")

    #train_dataset, validation_dataset, test_dataset = get_datasets(batch_size=batch_size)

    # create the base pre-trained model
    base_model = Xception(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(lmbda))(x)
    x = Dropout(dropout_rate)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    #plot_model(model, to_file='modelXception.png', show_shapes=True, show_layer_names=True)

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    learn_rate,
                    decay_steps=1000,
                    decay_rate=0.96,
                    staircase=True)


    if optimizer_name == 'Adagrad':
        optimizer = Adagrad(learning_rate=lr_schedule)
    elif optimizer_name == 'Adadelta':
        optimizer = Adadelta(learning_rate=lr_schedule)
    else:
        optimizer = Adam(learning_rate=lr_schedule)
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

    print('Start training...\n') 
    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_dataset,
                        verbose=0,
                        callbacks=[
                        WandbMetricsLogger(log_freq=5),
                        WandbModelCheckpoint("val_loss",
                                            save_weights_only=True, 
                                            save_best_only=True),
                        early_stopping
                        ])
    model.save_weights(os.path.join(wandb.run.dir, "model_weights.h5"))
    wandb.finish()

    test_evaluation = model.evaluate(test_dataset, verbose=0)
    print(f"Test evaluation: {test_evaluation[1]}")
    return max(history.history['val_accuracy'])
    #return test_evaluation



study = optuna.create_study(storage="sqlite:///c3_task3.db", 
                        study_name="100",
                        direction="maximize")

study.optimize(objective, n_trials=50)

# Print the best hyperparameters and result
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')