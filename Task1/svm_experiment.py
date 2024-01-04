import optuna 
import pickle
from sklearn.svm import SVC

train_images_filenames = pickle.load(open('../MIT_split/train_images_filenames.dat','rb'))
test_images_filenames = pickle.load(open('../MIT_split/test_images_filenames.dat','rb'))
train_images_filenames = ['..' + n[15:] for n in train_images_filenames]
test_images_filenames  = ['..' + n[15:] for n in test_images_filenames]
train_labels = pickle.load(open('../MIT_split/train_labels.dat','rb')) 
test_labels = pickle.load(open('../MIT_split/test_labels.dat','rb'))

visual_words = pickle.load(open('pickles/basic/visual_words.dat','rb'))
visual_words_test = pickle.load(open('pickles/basic/visual_words_test.dat','rb'))

def objective(trial):
    # Sample hyperparameters
    c = trial.suggest_categorical('C', [0.01, 0.1, 1, 10])
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', [0.01, 0.1, 1, 10])
    
    svm = SVC(C=c, kernel=kernel, gamma=gamma)
    svm.fit(visual_words, train_labels)
    return 100*svm.score(visual_words_test, test_labels)

search_space = {
    "C": [0.01, 0.1, 1, 10],
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "gamma": [0.01, 0.1, 1, 10],
}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space),
    direction="maximize",  # redundand, since grid search
    storage="sqlite:///svm.db",
    study_name="svm_experiment",
)
study.optimize(objective)
# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)