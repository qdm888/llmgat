import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _multiclass_acc(y_pred, y_true):
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


def eval_regression(y_pred, y_true, dataset):
    """Standard MSA metrics. Acc-2/F1 use the Non-0 protocol (y=0 excluded)."""
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    if dataset.lower() in ('mosi', 'mosei'):
        mae = float(np.mean(np.abs(y_pred - y_true)))
        corr = float(np.corrcoef(y_pred, y_true)[0][1])
        acc7 = _multiclass_acc(np.clip(y_pred, -3., 3.), np.clip(y_true, -3., 3.))
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        binary_truth = y_true[non_zeros] > 0
        binary_preds = y_pred[non_zeros] > 0
        return {
            'Non0_acc_2': round(float(accuracy_score(binary_preds, binary_truth)), 4),
            'Non0_F1_score': round(float(f1_score(binary_truth, binary_preds,
                                                  average='weighted')), 4),
            'Mult_acc_7': round(float(acc7), 4),
            'MAE': round(mae, 4),
            'Corr': round(corr, 4),
        }

    # CH-SIMS style: labels in [-1, 1]
    y_pred = np.clip(y_pred, -1., 1.)
    y_true = np.clip(y_true, -1., 1.)

    def bucketize(values, bounds):
        out = values.copy()
        for i in range(len(bounds) - 1):
            out[np.logical_and(values > bounds[i], values <= bounds[i + 1])] = i
        return out

    preds_a2 = bucketize(y_pred.copy(), [-1.01, 0.0, 1.01])
    truth_a2 = bucketize(y_true.copy(), [-1.01, 0.0, 1.01])
    preds_a3 = bucketize(y_pred.copy(), [-1.01, -0.1, 0.1, 1.01])
    truth_a3 = bucketize(y_true.copy(), [-1.01, -0.1, 0.1, 1.01])

    return {
        'Mult_acc_2': round(float(_multiclass_acc(preds_a2, truth_a2)), 4),
        'Mult_acc_3': round(float(_multiclass_acc(preds_a3, truth_a3)), 4),
        'F1_score': round(float(f1_score(truth_a2, preds_a2,
                                         average='weighted')), 4),
        'MAE': round(float(np.mean(np.abs(y_pred - y_true))), 4),
        'Corr': round(float(np.corrcoef(y_pred, y_true)[0][1]), 4),
    }
