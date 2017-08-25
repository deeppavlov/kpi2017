import sklearn.metrics

def roc_auc_score(y_true, y_pred):
    """ Compute Area Under the Curve (AUC) from prediction scores.
    y_true - true binary labels.
    y_pred - target scores, can either be probability estimates of the positive class.
    """

    try:
        return sklearn.metrics.roc_auc_score(y_true.reshape(-1), y_pred.reshape(-1))
    except ValueError:
        return 0.


