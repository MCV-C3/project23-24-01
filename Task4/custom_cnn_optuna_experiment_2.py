import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]=""

import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Input, BatchNormalization, MaxPooling2D
from keras.utils import plot_model
from keras.optimizers import Adam, Adagrad
from keras.utils.layer_utils import count_params

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


import optuna

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

wandb.finish()
wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'
IMG_WIDTH = 256
IMG_HEIGHT=256
NUMBER_OF_EPOCHS=50


def get_datasets(batch_size):
    train_data_generator = ImageDataGenerator(
    rescale=1./255    
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
        rescale=1./255
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
        rescale=1./255
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
    learning_rate = trial.suggest_float('learn_rate', 0.001, 0.1, log=True)
    K1 = trial.suggest_int("l1", 1, 64)
    K2 = trial.suggest_int("l2", 1, 64)
    K3 = trial.suggest_int("l3", 1, 64)
    K4 = trial.suggest_int("l4", 1, 64)
    K5 = trial.suggest_int("l5", 1, 64)

    wandb.init(project="final_K2", config=trial.params, name=f"run_{trial.number}")

    # create the base pre-trained model
    inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = Conv2D(K1, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(K2, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)

    x = Conv2D(K3, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(K4, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(K5, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(8, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    optimizer = Adam(learning_rate = learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.25, min_lr=1e-3)

    print('Start training...\n') 
    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_dataset,
                        verbose=0,
                        callbacks=[
                        WandbMetricsLogger(log_freq=5),
                        WandbModelCheckpoint(filepath=os.path.join(wandb.run.dir, "model_weights.h5"),
                                            monitor="val_loss",
                                            save_weights_only=True,
                                            save_best_only=True,
                                            mode='min'), 
                        early_stopping, 
                        reduce_lr_on_plateau
                        ])

    model.save_weights(os.path.join(wandb.run.dir, "model_weights.h5"))
    wandb.finish()

    test_evaluation = model.evaluate(test_dataset, verbose=0)
    print(f"Test evaluation: {test_evaluation[1]}")

    trainable_params = count_params(model.trainable_weights)
    print(f"Trainable: {trainable_params}")

    efficient_accuracy = max(history.history['val_accuracy']) * 100000 / trainable_params
    print(f"Efficient Accuracy: {efficient_accuracy}")

    print("Best trial number:", trial.number)

    return efficient_accuracy
    #return test_evaluation



study = optuna.create_study(storage="sqlite:///c3_task4.db", 
                        study_name="final_K2",
                        direction="maximize")

study.optimize(objective, n_trials=150)

# Print the best hyperparameters and result
print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print("Best trial number:", trial.number)
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')