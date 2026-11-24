"""Classification metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def metrics_for(y, probabilities, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(probabilities).reshape(-1)
    pred = (p >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0))
    }
