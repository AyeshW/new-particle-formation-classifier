import numpy as np
import json
import os
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, f1_score, brier_score_loss

def compute_binary_metrics(y_true, probs, eps=1e-15):
    probs = np.clip(probs, eps, 1-eps)
    preds = (probs > 0.5).astype(int)

    ll = log_loss(y_true, probs)
    acc = accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    f1 = f1_score(y_true, preds)
    brier = brier_score_loss(y_true, probs)
    perplexity = float(np.exp(ll))

    return {
        "log_loss": float(ll),
        "perplexity": float(perplexity),
        "accuracy": float(acc),
        "auc": float(auc),
        "f1": float(f1),
        "brier": float(brier)
    }

def multiclass_and_binary_metrics(y_true_class4, probs_class4, nonevent_label="nonevent", class_list=None):
    """
    y_true_class4 : array-like of true class4 labels (strings or ints)
    probs_class4  : ndarray (n_samples, n_classes) with prob columns ordered according to class_list
    class_list    : list of class labels in the same order as probs_class4 columns
    """
    eps = 1e-15
    probs = np.clip(probs_class4, eps, 1-eps)
    # map y_true to indices
    if class_list is None:
        raise ValueError("class_list must be provided")
    label_to_idx = {lab:i for i,lab in enumerate(class_list)}
    y_idx = np.array([label_to_idx[y] for y in y_true_class4])

    # multiclass log loss and accuracy (class4)
    mc_logloss = log_loss(y_idx, probs, labels=list(range(len(class_list))))
    y_pred_idx = np.argmax(probs, axis=1)
    class4_acc = accuracy_score(y_idx, y_pred_idx)

    # binary: event vs nonevent
    nonevent_idx = label_to_idx[nonevent_label]
    p_nonevent = probs[:, nonevent_idx]
    p_event = 1.0 - p_nonevent
    # binary true labels:
    y_binary = (y_idx != nonevent_idx).astype(int)
    # binary logloss
    bin_logloss = log_loss(y_binary, p_event)
    bin_acc = accuracy_score(y_binary, (p_event > 0.5).astype(int))
    perplexity = float(np.exp(bin_logloss))

    return {
        "multiclass_logloss": float(mc_logloss),
        "class4_accuracy": float(class4_acc),
        "binary_logloss": float(bin_logloss),
        "class2_accuracy": float(bin_acc),
        "perplexity": float(perplexity)
    }

def save_metrics(metrics_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

def get_aggregated_score(binary_accuracy, multiclass_accuracy, perplexity):
    """
    Compute aggregated score as the average of binary accuracy,
    multiclass accuracy, and the inverse of perplexity.
    """
    inv_perplexity = max(0, min(1, 2 - perplexity))
    aggregated_score = (binary_accuracy + multiclass_accuracy + inv_perplexity) / 3.0
    return aggregated_score
