import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
from keras import regularizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Dropout, BatchNormalization
from keras.utils import plot_model
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
from datetime import datetime
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from skimage.util import view_as_blocks

wandb.login(key="d1eed7aeb7e90a11c24c3644ed2df2d6f2b25718")

wandb.init(
    # set the wandb project where this run will be logged
    project="c3_project_patched",

    # track hyperparameters and run metadata with wandb.config
    config={"optimizer": "sgd",
            "loss": "categorical_crossentropy",
            "metric": "accuracy"}
)

config = wandb.config

current_datetime = datetime.now()

#user defined variables
for BATCH_SIZE in [32, 64, 128, 256]:
  for PATCH_SIZE in [32, 64, 128, 256]:
    DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
    PATCHES_DIR = '/ghome/group01/group01/project23-24-01/Task2/data/MIT_split_patches'+str(BATCH_SIZE)+"_"+str(PATCH_SIZE)
    MODEL_FNAME = f'/ghome/group01/group01/project23-24-01/Task2/weights/{BATCH_SIZE}_{PATCH_SIZE}_patch.weights.h5'


    if not os.path.exists(DATASET_DIR):
      print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
      quit()
    if not os.path.exists(PATCHES_DIR):
      print('WARNING: patches dataset directory '+PATCHES_DIR+' does not exist!\n')
      print('Creating image patches dataset into '+PATCHES_DIR+'\n')
      generate_image_patches_db(DATASET_DIR,PATCHES_DIR,patch_size=PATCH_SIZE)
      print('patxes generated!\n')

    # Data augmentation and preprocessing
    preprocessing_train = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
      tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
    ])

    preprocessing_validation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])

    # Load and preprocess the training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      directory=PATCHES_DIR+'/train/',
      labels='inferred',
      label_mode='categorical',
      batch_size=BATCH_SIZE,
      class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
      image_size=(PATCH_SIZE, PATCH_SIZE)
    )

    # Load and preprocess the validation dataset
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      directory=PATCHES_DIR+'/test/',
      labels='inferred',
      label_mode='categorical',
      batch_size=BATCH_SIZE,
      class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
      image_size=(PATCH_SIZE, PATCH_SIZE)
    )

    train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



    def build_mlp(input_size=PATCH_SIZE, phase='train'):
        model = Sequential()
        model.add(Reshape((input_size * input_size * 3,), input_shape=(input_size, input_size, 3)))
        model.add(Dense(units=2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.7))
        model.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        if phase == 'test':
            model.add(Dense(units=8, activation='linear'))
        else:
            model.add(Dense(units=8, activation='softmax'))

        return model

    print('Building MLP model...\n')

    model = build_mlp(input_size=PATCH_SIZE)

    model.compile(loss=config.loss,
                  optimizer=config.optimizer,
                  metrics=[config.metric])

    print(model.summary())


    train = False
    if  not os.path.exists(MODEL_FNAME) or train:
      print('WARNING: model file '+MODEL_FNAME+' do not exists!\n')
      print('Start training...\n')
      
      history = model.fit(train_dataset,
                epochs=150,
                validation_data=validation_dataset,
                verbose=0,
                callbacks=[
                          WandbMetricsLogger(log_freq=5),
                          WandbModelCheckpoint("models"),
                        ])
      print('Saving the model into W&B \n')
      model.save(os.path.join(wandb.run.dir, "model.h5"))
      wandb.finish()

      print('Saving the model into '+MODEL_FNAME+' \n')
      model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
      print('Done!\n')

    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'/ghome/group01/group01/project23-24-01/Task2/results/mlp_patch/{BATCH_SIZE}_{PATCH_SIZE}_accuracy_patch.jpg')
    plt.close()

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'/ghome/group01/group01/project23-24-01/Task2/results/mlp_patch/{BATCH_SIZE}_{PATCH_SIZE}_loss_patch.jpg')
    plt.close()

    print('Building MLP model for testing...\n')

    model = build_mlp(input_size=PATCH_SIZE, phase='test')
    print(model.summary())

    print('Done!\n')

    print('Loading weights from '+MODEL_FNAME+' ...\n')
    print ('\n')

    model.load_weights(MODEL_FNAME)

    print('Done!\n')

    print('Start evaluation ...\n')

    directory = DATASET_DIR+'/test'
    classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
    correct = 0.
    total   = 807
    count   = 0

    for class_dir in os.listdir(directory):
        cls = classes[class_dir]
        for imname in os.listdir(os.path.join(directory,class_dir)):
          im = Image.open(os.path.join(directory,class_dir,imname))
          # patches = view_as_blocks(np.array(im), block_shape=(PATCH_SIZE, PATCH_SIZE, 3)).reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
          patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE), max_patches=1)
          out = model.predict(patches/255.)
          predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )
          if predicted_cls == cls:
            correct+=1
          count += 1
          print('Evaluated images: '+str(count)+' / '+str(total), end='\r')
        
    print('Done!\n')
    print('Test Acc. = '+str(correct/total)+'\n')
