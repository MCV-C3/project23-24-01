import os
import optuna
import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

train_images_filenames = pickle.load(open('train_images_filenames.dat','rb'))
test_images_filenames = pickle.load(open('test_images_filenames.dat','rb'))
train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
test_images_filenames  = ['..' + n[15:] for n in test_images_filenames]
train_labels = pickle.load(open('train_labels.dat','rb')) 
test_labels = pickle.load(open('test_labels.dat','rb'))


for file in range(len(train_images_filenames)):
    train_images_filenames[file] = train_images_filenames[file][3:]
for file in range(len(test_images_filenames)):
    test_images_filenames[file] = test_images_filenames[file][3:]

Detector = cv2.SIFT_create(nfeatures=1750)

def dense_sift(img, keypoint_size=14, h_size=40, w_size=30):
    # Compute SIFT descriptors at each grid point
    h, w = img.shape
    margin = int(keypoint_size/2)
    keypoints = [cv2.KeyPoint(x, y, keypoint_size) for y in range(margin, h, keypoint_size) for x in range(margin, w, keypoint_size)]
    _, descriptors = Detector.compute(img, keypoints)

    return keypoints, descriptors

def dense_sift_v2(img, keypoint_size=16, h_size=40, w_size=30):
    # Compute SIFT descriptors at each grid point
    h, w = img.shape
    margin = int(keypoint_size/2)
    step_size_h = int(h/h_size)
    step_size_w = int(w/w_size)
    keypoints = [cv2.KeyPoint(x, y, keypoint_size) for y in range(margin, h - margin, step_size_h) for x in range(margin, w - margin, step_size_w)]
    _, descriptors = Detector.compute(img, keypoints)

    return keypoints, descriptors

def objective(trial):
    #nfeatures = trial.suggest_int("nfeatures", 0, 20000)
    kpt_size = trial.suggest_int("kpt_size", 1, 100)
    #h_size = trial.suggest_int("h_size", 1, 50)
    #w_size = trial.suggest_int("w_size", 1, 50)
    
    Train_descriptors = []
    Train_label_per_descriptor = []

    for filename,labels in zip(train_images_filenames,train_labels):
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        #kpt,des=Detector.detectAndCompute(gray,None)
        kpt,des = dense_sift_v2(gray, keypoint_size=kpt_size)
        #des = np.nan_to_num(des, nan=0.0)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(labels)

    D=np.vstack(Train_descriptors)

    k = 128
    codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
    codebook.fit(D)

    visual_words=np.zeros((len(Train_descriptors),k),dtype=np.float32)
    for i in range(len(Train_descriptors)):
        words=codebook.predict(Train_descriptors[i])
        visual_words[i,:]=np.bincount(words,minlength=k)

    knn = KNeighborsClassifier(n_neighbors=8,n_jobs=-1,metric='euclidean')
    knn.fit(visual_words, train_labels)

    visual_words_test=np.zeros((len(test_images_filenames),k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        filename=test_images_filenames[i]
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        #kpt,des=Detector.detectAndCompute(gray,None)
        kpt,des = dense_sift(gray, keypoint_size=kpt_size,)
        #des = np.nan_to_num(des, nan=0.0)
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k) 

    accuracy = 100*knn.score(visual_words_test, test_labels)

    return accuracy

"""
search_space = {
    "kpt_size": [12, 14, 16],
    "h_size": [30, 40],
    "w_size": [30, 40]
}
"""



search_space = {
    "kpt_size": [6, 8, 12, 14, 16, 18, 20, 25, 32, 64],
    #"nfeatures": [0, 750, 1000, 1250, 1500, 1750, 2000]
}

sampler = optuna.samplers.GridSampler(search_space)

study = optuna.create_study(storage="sqlite:///c3_task1.db", 
                        study_name="siftV4",
                        sampler=sampler,
                        direction="maximize")

study.optimize(objective)

# trial: single combination of parameters + run of objective func
# study: log book / collection of trials
