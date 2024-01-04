import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def display_multilabel_confusion_matrix(visual_words_test, test_labels, model):
    cm = multilabel_confusion_matrix(model.predict(visual_words_test), test_labels)

    _, axes = plt.subplots(2, 4, figsize=(15, 15))
    for i, (matrix, ax) in enumerate(zip(cm, axes.flatten())):
        class_name = f"Class {i}"
        labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
        labels = [f"{class_name}\n{label}" for label in labels]
        
        ax.imshow(matrix, cmap='Blues', interpolation='nearest')
        ax.set(xticks=np.arange(2), yticks=np.arange(2), xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
        ax.set_title(f"Confusion Matrix for {class_name}")

        # Display the values inside the heatmap
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()
    
def fit_predict_and_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    (f1, accuracy, precision, recall) = \
        (f1_score(y_test, y_pred, average='macro'), 
        accuracy_score(y_test, y_pred), 
        precision_score(y_test, y_pred, average='macro'), 
        recall_score(y_test, y_pred, average='macro'))
    print(f"    -> F1-Score: {f1*100:.2f}")
    print(f"    -> Accuracy: {accuracy*100:.2f}")
    print(f"    -> Precision: {precision*100:.2f}")
    print(f"    -> Recall: {recall*100:.2f}")