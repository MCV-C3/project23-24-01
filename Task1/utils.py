import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

def display_multilabel_confusion_matrix(visual_words_test, test_labels, model):
    cm = multilabel_confusion_matrix(model.predict(visual_words_test), test_labels)

    _, axes = plt.subplots(1, cm.shape[0], figsize=(70, 15))
    for i, matrix in enumerate(cm):
        class_name = f"Class {i}"
        labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
        labels = [f"{class_name}\n{label}" for label in labels]
        
        ax = axes[i]
        ax.imshow(matrix, cmap='Blues', interpolation='nearest')
        ax.set(xticks=np.arange(2), yticks=np.arange(2), xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
        ax.set_title(f"Confusion Matrix for {class_name}")

        # Display the values inside the heatmap
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')

    plt.show()