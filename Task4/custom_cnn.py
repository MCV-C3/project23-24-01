from keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt

MODEL_NAME = 'model_name.h5'
WEIGHTS_DIR = f'/ghome/group01/group01/project23-24-01/Task4/weights/{MODEL_NAME}'
RESULTS_DIR = '/ghome/group01/group01/project23-24-01/Task4/results'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'

NUM_CLASSES = 8
BATCH_SIZE = 64
IMG_SIZE = (64, 64)
EPOCHS = 250

def get_datasets():
    train_data_generator = ImageDataGenerator(
        rescale=1./255    
    )

    # Load and preprocess the training dataset
    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR+'/train/',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the validation dataset
    validation_data_generator = ImageDataGenerator(
        rescale=1./255
    )

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=DATASET_DIR+'/test/',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the test dataset
    test_data_generator = ImageDataGenerator(
        rescale=1./255
    )

    test_dataset = test_data_generator.flow_from_directory(
        directory=DATASET_DIR_GLOBAL+'/test/',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    return train_dataset, validation_dataset, test_dataset

def build_model():
    model = Sequential()
    # Input layer
    model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # # Second convolutional layer
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # # Third convolutional layer
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # # Flatten the output before feeding into dense layers
    # model.add(Flatten())

    # # Dense layers with dropout for regularization
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def plot_metrics(history):
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{RESULTS_DIR}/model_accuracy.jpg')
    plt.close()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{RESULTS_DIR}/model_loss.jpg')
    plt.close()

train_dataset, validation_dataset, test_dataset = get_datasets()
model = build_model()

print(model.summary())
plot_model(model, to_file=RESULTS_DIR / 'model.png', show_shapes=True, show_layer_names=True)

print('Start training...\n')
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    verbose=0,
    callbacks=[]
)

model.save_weights(WEIGHTS_DIR)
plot_metrics(history)