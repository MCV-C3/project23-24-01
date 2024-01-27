from keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Add, Input
from keras.models import Model
import matplotlib.pyplot as plt

MODEL_NAME = 'model_cpu_v2'
WEIGHTS_DIR = f'/ghome/group01/group01/project23-24-01/Task4/weights/{MODEL_NAME}.h5'
RESULTS_DIR = '/ghome/group01/group01/project23-24-01/Task4/results'
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_small_train_1'
DATASET_DIR_GLOBAL = '/ghome/mcv/datasets/C3/MIT_split'

NUM_CLASSES = 8
BATCH_SIZE = 64
IMG_SIZE = (256, 256)
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
    num_kernels = 16
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Initial Convolution Block
    x = Conv2D(num_kernels, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)

    # Residual Blocks
    for _ in range(2):
        residual = x
        x = Conv2D(2*num_kernels, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        x = Conv2D(2*num_kernels, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.7)(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = Conv2D(2*num_kernels, (1, 1), activation='relu', padding='same')(residual)

        x = Add()([x, residual])
        x = MaxPooling2D((3, 3))(x)

    # Global Average Pooling and Dense Layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.7)(x)

    # Output layer
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=predictions)

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
    plt.savefig(f'{RESULTS_DIR}/{MODEL_NAME}_accuracy.jpg')
    plt.close()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{RESULTS_DIR}/{MODEL_NAME}_loss.jpg')
    plt.close()

train_dataset, validation_dataset, test_dataset = get_datasets()
model = build_model()

print(model.summary())
plot_model(model, to_file=f'{RESULTS_DIR}/{MODEL_NAME}.png', show_shapes=True, show_layer_names=True)

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