import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Input, BatchNormalization, MaxPooling2D, Flatten
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.utils.layer_utils import count_params
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import optuna
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

wandb.finish()
wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUMBER_OF_EPOCHS = 50

def get_datasets(batch_size):
    train_data_generator = ImageDataGenerator(
        rescale=(1. / 255) * 2 - 1,
    )

    # Load and preprocess the training dataset
    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/train/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the validation dataset
    validation_data_generator = ImageDataGenerator(
        rescale=(1. / 255) * 2 - 1
    )

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the test dataset
    test_data_generator = ImageDataGenerator(
        rescale=(1. / 255) * 2 - 1
    )

    test_dataset = test_data_generator.flow_from_directory(
        directory=DATASET_DIR_GLOBAL + '/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    return train_dataset, validation_dataset, test_dataset


wandb.finish()
train_dataset, validation_dataset, test_dataset = get_datasets(batch_size=16)

wandb.init(project="test_aux", config={'learning_rate': 0.01})
config = wandb.config

# Define boolean flags for each layer
use_layer_1 = True
use_layer_2 = True
use_layer_3 = True
use_layer_4 = True
use_layer_5 = True
use_layer_6 = True
use_layer_7 = True

# create the base pre-trained model
inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = inputs

# Add layer 1
if use_layer_1:
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

# Add layer 2
if use_layer_2:
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

# Add layer 3
if use_layer_3:
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

# Add layer 4
if use_layer_4:
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

# Add layer 5
if use_layer_5:
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

# Add layer 6
if use_layer_6:
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

# Add layer 7
if use_layer_7:
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(8, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

print(model.summary())

optimizer = Adam(learning_rate=config.learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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

# Prune the model
num_images = 400
batch_size = 16
epochs = NUMBER_OF_EPOCHS

num_images = train_dataset.samples
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_schedule = sparsity.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=0.5,  # Adjust the final sparsity as needed
    begin_step=0,
    end_step=end_step  # You need to define end_step, as shown in your previous code
)

pruning_model = sparsity.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

# Compile the pruned model
pruning_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the pruned model if necessary

# Evaluate the pruned model
pruning_model.evaluate(test_dataset)

test_evaluation = pruning_model.evaluate(test_dataset, verbose=0)
print(f"Pruned Test evaluation: {test_evaluation[1]}")
