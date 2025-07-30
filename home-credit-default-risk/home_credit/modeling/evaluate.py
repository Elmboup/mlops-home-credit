import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, metrics, average="binary"):
    y_pred = model.predict(X_test)
    results = {}
    for name, metric in metrics.items():
        if name == "roc_auc":
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                results[name] = metric(y_test, y_score)
            else:
                results[name] = None
        elif name in ["f1", "precision", "recall"]:
            results[name] = metric(y_test, y_pred, average=average)
        else:
            results[name] = metric(y_test, y_pred)
    return results


def plot_roc_curve(model, X_test, y_test, ax=None):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Le modèle ne supporte pas predict_proba.")
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(model, X_test, y_test, ax=None):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Le modèle ne supporte pas predict_proba.")
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(recall, precision, color='blue', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, ax=None, labels=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show() 