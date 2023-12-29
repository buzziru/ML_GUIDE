import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import Binarizer


import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_clf_eval(y_test, pred, pred_proba=None):
    """
+    Calculate evaluation metrics for a classification model.
+
+    Args:
+        y_test (array-like): The true labels of the test set.
+        pred (array-like): The predicted labels of the test set.
+        pred_proba (array-like, optional): The predicted probabilities of the test set. Defaults to None.
+
+    Returns:
+        None
+
+    Raises:
+        None
+    """
    # 이진 분류와 다중 클래스 분류를 구분하기 위한 클래스 개수 확인
    class_count = len(np.unique(y_test))

    # 이진 분류인 경우
    if class_count == 2:
        average_type = 'binary'
    # 다중 클래스 분류인 경우
    else:
        average_type = 'macro'

    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=average_type)
    recall = recall_score(y_test, pred, average=average_type)
    f1 = f1_score(y_test, pred, average=average_type)
    
    if pred_proba is not None:
        if class_count == 2:
            roc_score = roc_auc_score(y_test, pred_proba)
        else:
            roc_score = roc_auc_score(y_test, pred_proba, average=average_type)
        print('ROC AUC:', round(roc_score, 4))
    
    print('Confusion Matrix')
    print(confusion)
    print('Accuracy:', round(accuracy, 4), 'Precision:', round(precision, 4), 
          'Recall:', round(recall, 4))
    print('F1:', round(f1, 4))


def get_eval_by_threshold(y_test, pred_proba, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba)
        custom_predict = binarizer.transform(pred_proba)
        print('THRESHOLD : ', custom_threshold)
        get_clf_eval(y_test, custom_predict, pred_proba)
        print('='*50)


def roc_curve_plot(y_test, pred_proba_c1):
    """
    Generate a ROC curve plot based on the predicted probabilities and true labels.
    Parameters:
    - y_test (array-like): The true labels of the binary classification problem.
    - pred_proba_c1 (array-like): The predicted probabilities of the positive class.
    Returns:
    - None
    This function calculates the false positive rates (FPRs), true positive rates (TPRs),
    and thresholds using the roc_curve function. It also calculates the ROC score using
    the roc_auc_score function. The FPRs, TPRs, and thresholds are then plotted on a
    graph using the plot function from the Matplotlib library. The function also adds a
    line representing the random classifier. The x-axis represents the FPR (1 -
    Sensitivity) and the y-axis represents the TPR (Recall). The graph is displayed
    using the show function from the Matplotlib library. The ROC score is printed
    rounded to four decimal places.
    Note: This function requires the following libraries to be imported:
    - roc_curve from sklearn.metrics
    - roc_auc_score from sklearn.metrics
    - plt from matplotlib.pyplot
    - np from numpy
    """
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    roc_score = roc_auc_score(y_test, pred_proba_c1)
    plt.plot(fprs, tprs, label='ROC')
    plt.plot([0, 1], [0, 1], 'k-', label='Random')

    plt.xlim((0, 1))
    plt.ylabel((0, 1))
    plt.xticks(np.round(np.arange(0.1, 1.0, 0.1), 2))
    plt.xlabel('FPR(1 - Sensitivity)')
    plt.ylabel('TPR(Recall)')
    plt.legend()
    plt.show()

    print('ROC SCORE : ', round(roc_score, 4))


def precision_recall_curve_plot(y_test, pred_proba_c1):
    """
    Generates a precision-recall curve plot based on the predicted probabilities of class 1 and the true labels.

    Parameters:
        y_test (array-like): The true labels.
        pred_proba_c1 (array-like): The predicted probabilities of class 1.

    Returns:
        None
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)
    
    # set threshold into X-axis and precision, recall into Y-axis
    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[:threshold_boundary], ls='--', label='precision')
    plt.plot(thresholds, recalls[:threshold_boundary], ls='-', label='recall')
    
    plt.xlim((0.1, 0.9))
    plt.xticks(np.round(np.arange(0.1, 0.9, 0.1), 2))

    plt.xlabel('Threshold value')
    plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.show()
