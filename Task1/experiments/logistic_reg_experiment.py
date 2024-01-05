import optuna 
import cv2
import numpy as np
import pickle
import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_images_filenames = pickle.load(open('../MIT_split/train_images_filenames.dat','rb'))
test_images_filenames = pickle.load(open('../MIT_split/test_images_filenames.dat','rb'))
train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
test_images_filenames  = ['..' + n[15:] for n in test_images_filenames]
train_labels = pickle.load(open('../MIT_split/train_labels.dat','rb')) 
test_labels = pickle.load(open('../MIT_split/test_labels.dat','rb'))
def objective(trial):
    # Sample hyperparameters
    k_means_clusters = trial.suggest_int('k_means_clusters', 64, 384, step=64)
    
    #Detector = cv2.KAZE_create(threshold=0.0001)
    Detector = cv2.SIFT_create(nfeatures=2000)

    Train_descriptors = []
    Train_label_per_descriptor = []
    # Create a label encoder
    label_encoder = LabelEncoder()

    # Encode the train labels
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    for filename,labels in zip(train_images_filenames,train_labels_encoded):
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt,des=Detector.detectAndCompute(gray,None)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(labels)

    D=np.vstack(Train_descriptors)
    
    # K-Means clustering
    codebook = MiniBatchKMeans(n_clusters=k_means_clusters, verbose=False, batch_size=k_means_clusters * 20,
                               compute_labels=False, reassignment_ratio=10**-4, random_state=42)
    codebook.fit(D)

    visual_words = np.zeros((len(Train_descriptors), k_means_clusters), dtype=np.float32)
    for i in range(len(Train_descriptors)):
        words = codebook.predict(Train_descriptors[i])
        visual_words[i, :] = np.bincount(words, minlength=k_means_clusters)
    visual_words_test=np.zeros((len(test_images_filenames),k_means_clusters),dtype=np.float32)
    visual_words_dense_test=np.zeros((len(test_images_filenames),k_means_clusters),dtype=np.float32)

    for i in range(len(test_images_filenames)):
        filename=test_images_filenames[i]
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt,des=Detector.detectAndCompute(gray,None)
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k_means_clusters)


    # Create a logistic regression model
    logistic_reg = LogisticRegression(max_iter=10000)  

    # Convert string labels to numerical labels
    label_mapping = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    train_labels_numeric = np.array([label_mapping[label] for label in train_labels])

    # Train the model
    logistic_reg.fit(visual_words, train_labels_numeric)

    # Make predictions on the test data
    predictions_numeric = logistic_reg.predict(visual_words_test)
    # Assuming you have a label_mapping dictionary that maps numerical labels to string labels
    label_mapping_inverse = {idx: label for label, idx in label_mapping.items()}
    # Convert numerical predictions back to string labels
    predictions = np.array([label_mapping_inverse[idx] for idx in predictions_numeric])
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

search_space = {
    "k_means_clusters": [64, 128, 192, 256, 320, 384],
}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///hparam.db",
    study_name="l1",
)
study.optimize(objective)
# Get the best hyperparameters
best_params = study.best_params
print("Best number of clusters:", best_params)