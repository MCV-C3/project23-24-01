import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize

def display_multilabel_confusion_matrix(visual_words_test, test_labels, model, mapping=None):
    if mapping is not None:
        predictions_numeric = model.predict(visual_words_test)
        label_mapping_inverse = {idx: label for label, idx in mapping.items()}
        predictions = np.array([label_mapping_inverse[idx] for idx in predictions_numeric])
        cm = multilabel_confusion_matrix(predictions, test_labels)
    else:
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
    print(f"    -> F1-Score: {f1*100:.2f}%")
    print(f"    -> Accuracy: {accuracy*100:.2f}%")
    print(f"    -> Precision: {precision*100:.2f}%")
    print(f"    -> Recall: {recall*100:.2f}%")
    
def plot_roc_curve(visual_words_test, test_labels, model, model_name):
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(test_labels)

    # Binarize the labels
    y_test_bin = label_binarize(y_test_encoded, classes=np.arange(8))

    # Get the predicted probabilities for each class
    if model_name == "SVM":
        y_score = model.decision_function(visual_words_test)
    else:
        y_score = model.predict_proba(visual_words_test)

    # Initialize the ROC curve plot
    plt.figure(figsize=(10, 6))

    # Compute ROC curve and ROC area for each class
    for i in range(8):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Retrieve the original string label
        original_label = label_encoder.inverse_transform([i])[0]
        
        # Plot the ROC curve for each class with original label
        plt.plot(fpr, tpr, label=f'{original_label} (AUC = {roc_auc:.2f})')

    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()