"""Shard learners: random forest, neural network and logistic regression (SGD)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------
# Shard learners
#   "sgd"    : low-capacity linear model, trained incrementally per slice.
#   "forest" : fully grown random forest. It memorises its training rows,
#              which creates a genuine membership signal for the MIA demo.
# ---------------------------------------------------------------------
def calculate_class_weights(y):
    """Compute balanced weights explicitly for SGDClassifier.partial_fit().

    sklearn does not accept class_weight="balanced" with partial_fit.
    Classes absent from a shard receive weight 1.0; present classes are
    inversely weighted by their frequency within that shard.
    """
    y = np.asarray(y, dtype=int)
    total = len(y)
    weights = {}
    for class_id in (0, 1):
        count = int(np.sum(y == class_id))
        weights[class_id] = float(total / (2 * count)) if count else 1.0
    return weights


def make_sgd(class_weights):
    return SGDClassifier(
        loss="log_loss", alpha=0.0005, max_iter=1,
        learning_rate="optimal", random_state=42,
        class_weight=class_weights, tol=None
    )


def make_mlp():
    # High-capacity parametric model: wide, barely regularised, trained for many passes.
    # Unlike the linear SGD model it can fit individual training rows, so it leaks membership.
    return MLPClassifier(
        hidden_layer_sizes=(256, 256), alpha=1e-6, batch_size=256,
        learning_rate_init=1e-3, max_iter=1, shuffle=True, random_state=42
    )


def make_incremental(model_type, y):
    """Parametric shard learner trained with partial_fit (supports SISA slices)."""
    if model_type == "mlp":
        return make_mlp()
    return make_sgd(calculate_class_weights(y))


def make_forest(n_trees=100):
    # Every tree sees ALL of the shard's rows (no bootstrap) and is grown until pure, so each
    # tree fits every training record exactly. Members therefore get a crisp, near-perfect
    # response, which gives the membership attack a signal for (almost) every training patient,
    # and unlearning has a measurable effect on each of them.
    return RandomForestClassifier(
        n_estimators=int(n_trees), bootstrap=False, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", n_jobs=-1, random_state=42
    )


def positive_proba(clf, X):
    """P(class 1), robust to a shard model that only ever saw one class."""
    p = clf.predict_proba(X)
    if p.shape[1] == 1:
        return p[:, 0] if int(clf.classes_[0]) == 1 else np.zeros(len(X))
    return p[:, list(clf.classes_).index(1)]
