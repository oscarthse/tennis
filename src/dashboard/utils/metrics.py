from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for the positive class (1)

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Brier Score': brier_score_loss(y_true, y_prob)
    }

    try:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['ROC-AUC'] = None

    try:
        metrics['Log Loss'] = log_loss(y_true, y_prob)
    except ValueError:
        metrics['Log Loss'] = None

    return metrics

def get_confusion_matrix(y_true, y_pred):
    """Return confusion matrix."""
    return confusion_matrix(y_true, y_pred)
