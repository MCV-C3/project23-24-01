from __future__ import print_function
from utils import *
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator

import pickle
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#user defined variables
for BATCH_SIZE in [32, 64, 128, 256]:
  for PATCH_SIZE in [32, 64, 128, 256]:
        IMG_SIZE = 256
        DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
        MODEL_FNAME = f'/ghome/group01/group01/project23-24-01/Task2/models/patches/{BATCH_SIZE}_{PATCH_SIZE}.h5'
        RESULTS = f'/ghome/group01/group01/project23-24-01/Task2/results/mlp_bovw_svm/'


        train_images_filenames = pickle.load(open('/ghome/group01/group01/project23-24-01/Task2/data/train_images_filenames.dat','rb'))
        test_images_filenames = pickle.load(open('/ghome/group01/group01/project23-24-01/Task2/data/test_images_filenames.dat','rb'))
        train_labels = pickle.load(open('/ghome/group01/group01/project23-24-01/Task2/data/train_labels.dat','rb'))
        test_labels = pickle.load(open('/ghome/group01/group01/project23-24-01/Task2/data/test_labels.dat','rb'))

        model = load_model(MODEL_FNAME)
        model = Model(inputs=model.input, outputs=model.layers[-2].output)
        model.summary()

        PATCH_SIZE = model.layers[0].input.shape[1:3]
        NUM_PATCHES = (IMG_SIZE//PATCH_SIZE.as_list()[0])**2

        codebook_size = 512 # 760

        Train_descriptors = []
        Train_label_per_descriptor = []

        def get_descriptors(model, images_filenames):
            descriptors = np.empty((len(images_filenames), NUM_PATCHES, model.layers[-1].output_shape[1]))
            for i,filename in enumerate(images_filenames):
                img = Image.open(filename)
                patches = image.extract_patches_2d(np.array(img), PATCH_SIZE, max_patches=NUM_PATCHES)
                descriptors[i, :, :] = model.predict(patches/255.)

            return descriptors

        def get_visual_words(descriptors, codebook, codebook_size):
            visual_words=np.empty((len(descriptors),codebook_size),dtype=np.float32)
            for i,des in enumerate(descriptors):
                words=codebook.predict(des)
                visual_words[i,:]=np.bincount(words,minlength=codebook_size)

            return StandardScaler().fit_transform(visual_words)

        train_descriptors = get_descriptors(model, train_images_filenames)

        codebook = MiniBatchKMeans(n_clusters=codebook_size,
                                    verbose=False,
                                    batch_size=codebook_size * 20,
                                    compute_labels=False,
                                    reassignment_ratio=10**-4,
                                    random_state=42)

        codebook.fit(np.vstack(train_descriptors))

        train_visual_words = get_visual_words(train_descriptors, codebook, codebook_size)

        # gridsearch SVM
        parameters = {'kernel': ('rbf', 'linear', 'sigmoid')}
        grid = GridSearchCV(svm.SVC(), parameters, n_jobs=3, cv=8)
        grid.fit(train_visual_words, train_labels)

        best_kernel = grid.best_params_['kernel']
        print(f'Best SVM kernel: {best_kernel}')

        # test
        test_descriptors = get_descriptors(model, test_images_filenames)
        test_visual_words = get_visual_words(test_descriptors, codebook, codebook_size)

        classifier = svm.SVC(kernel=best_kernel)
        classifier.fit(train_visual_words,train_labels)

        roc_curve(test_visual_words, test_labels, classifier, RESULTS+f'ROC_{BATCH_SIZE}_{PATCH_SIZE}.png')

        accuracy = classifier.score(test_visual_words, test_labels)

        print(f'Test accuracy: {accuracy}')

        display_multilabel_confusion_matrix(test_visual_words, test_labels, classifier, RESULTS+f'confusion_matrix_ROC_{BATCH_SIZE}_{PATCH_SIZE}.png')
