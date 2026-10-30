"""Explicit replacement CV protocol, not a claim about undocumented author CV."""
import numpy as np

from .model import _data


class TrainPreprocessor:
    """Fit state only on training rows; raw/l2/standard are explicit alternatives."""
    def __init__(self, mode="raw"):
        if mode not in ("raw", "l2", "standard"):
            raise ValueError("Unknown preprocessing mode")
        self.mode = mode

    def fit(self, X):
        X = _data(X)
        self.mean_ = X.mean(axis=0) if self.mode == "standard" else np.zeros(X.shape[1])
        scale = X.std(axis=0) if self.mode == "standard" else np.ones(X.shape[1])
        self.scale_ = np.where(scale > np.finfo(float).eps, scale, 1.)
        return self

    def transform(self, X):
        X = _data(X)
        if X.shape[1] != len(self.mean_):
            raise ValueError("Feature dimension differs from training")
        if self.mode == "l2":
            return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), np.finfo(float).eps)
        return (X - self.mean_) / self.scale_

    def save(self, path):
        from .checkpoints import atomic_npz
        atomic_npz(path, mode=np.array(self.mode), mean=self.mean_, scale=self.scale_)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(str(data["mode"]))
            model.mean_, model.scale_ = data["mean"].copy(), data["scale"].copy()
        return model


def select_candidate(scores):
    """Maximize unweighted mean validation ACC, exact ties use declared order.

    All candidates must have every fold. A failed or nonfinite fold is an error,
    not a missing value to drop or a reason to choose from successful survivors.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("Every candidate must have a finite score for every fold")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("ACC must lie in [0,1]")
    means = scores.mean(axis=1)
    return int(np.argmax(means)), means
