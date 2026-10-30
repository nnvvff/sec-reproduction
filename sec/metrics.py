"""Permutation-invariant clustering ACC, Eq. (35), and frozen train mapping."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def fit_label_mapping(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or len(y_true) == 0:
        raise ValueError("Expected nonempty equally sized 1D label arrays")
    classes, inverse_true = np.unique(y_true, return_inverse=True)
    clusters, inverse_pred = np.unique(y_pred, return_inverse=True)
    counts = np.zeros((len(clusters), len(classes)), dtype=int)
    np.add.at(counts, (inverse_pred, inverse_true), 1)
    rows, cols = linear_sum_assignment(-counts)
    return {clusters[r].item(): classes[c].item() for r, c in zip(rows, cols)}


def mapped_accuracy(y_true, y_pred, mapping):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or not len(y_true):
        raise ValueError("Expected nonempty equally sized 1D label arrays")
    return float(np.mean([p in mapping and mapping[p] == t for t, p in zip(y_true, y_pred)]))


def clustering_accuracy(y_true, y_pred):
    return mapped_accuracy(y_true, y_pred, fit_label_mapping(y_true, y_pred))
