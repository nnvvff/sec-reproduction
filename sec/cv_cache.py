"""Reuse training-fold matrices across SEC parameters; no test data enters the cache."""
import warnings
import numpy as np
from scipy import linalg
from .model import (_data, SpectralEmbeddedClustering, local_regression_laplacian,
                    sc_laplacian, spectral_rotation)


class FoldCache:
    """Exact SEC computations on one raw-feature training fold."""

    def __init__(self, X_train):
        self.X = np.array(_data(X_train), dtype=float, copy=True, order="C")
        self.X.flags.writeable = False
        self.mean = self.X.mean(axis=0)
        self.n, self.d = self.X.shape
        self.stats = {"svd_once": 0, "svd_twice": 0, "global_laplacian": 0,
                      "sc_laplacian": 0, "lr_laplacian": 0}
        self._centered = None
        self._svds, self._global, self._sc, self._lr = {}, {}, {}, {}

    def fit(self, model):
        if type(model) is not SpectralEmbeddedClustering:
            raise TypeError("FoldCache supports SEC only")
        if not isinstance(model.n_clusters, (int, np.integer)) or not 1 <= model.n_clusters < self.n:
            raise ValueError("n_clusters must be in [1, n_samples-1]")
        self._positive(model.gamma_global, "gamma_global")
        return self._fit_sec(model)

    def _Z(self):
        if self._centered is None:
            self._centered = self.X - self.mean
            self._centered.flags.writeable = False
        return self._centered


    def _svd(self, twice=False):
        if twice not in self._svds:
            Z = self._Z()
            data = Z - Z.mean(axis=0) if twice else Z
            self._svds[twice] = linalg.svd(data, full_matrices=False, check_finite=False)
            self.stats["svd_twice" if twice else "svd_once"] += 1
        return self._svds[twice]


    @staticmethod
    def _positive(value, name):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


    def _global_laplacian(self, gamma):
        self._positive(gamma, "gamma_global")
        if gamma not in self._global:
            U, s, _ = self._svd(twice=True)
            shrink = s**2 / (s**2 + gamma)
            Lg = np.eye(self.n) - np.full((self.n, self.n), 1 / self.n)
            Lg -= (U * shrink) @ U.T
            self._global[gamma] = (Lg + Lg.T) / 2
            self.stats["global_laplacian"] += 1
        return self._global[gamma]


    def _sc_laplacian(self, model):
        key = (model.n_neighbors, model.scale_neighbor)
        if key not in self._sc:
            self._sc[key] = sc_laplacian(self._Z(), *key)
            self.stats["sc_laplacian"] += 1
        return self._sc[key]


    def _lr_laplacian(self, model):
        key = (model.n_neighbors, model.gamma_local, model.lr_include_self)
        if key not in self._lr:
            self._lr[key] = local_regression_laplacian(self._Z(), *key)
            self.stats["lr_laplacian"] += 1
        return self._lr[key]


    def _rotate(self, model, F):
        result = spectral_rotation(F, model.n_init, model.random_state, model.max_iter)
        model.labels_ = result.labels
        model.rotation_ = result.rotation
        model.rotation_objective_ = result.objective
        model.rotation_history_ = result.history
        model.rotation_converged_ = result.converged
        model.n_clusters_found_ = len(np.unique(model.labels_))


    def _ridge(self, model, F, twice=False):
        U, s, Vt = self._svd(twice=twice)
        model.mean_ = self.mean.copy()
        model.n_features_in_ = self.d
        model.coef_ = (Vt.T * (s / (s*s + model.gamma_global))) @ (U.T @ F)
        model.embedding_bias_ = F.mean(axis=0)
        model.intercept_ = model.embedding_bias_ - model.mean_ @ model.coef_


    def _fit_sec(self, model):
        if model.variant not in ("sc", "lr"):
            raise ValueError("variant must be 'sc' or 'lr'")
        if not np.isfinite(model.mu) or model.mu < 0:
            raise ValueError("mu must be finite and nonnegative")
        L = self._sc_laplacian(model) if model.variant == "sc" else self._lr_laplacian(model)
        model.zero_local_diagonal_count_ = int(np.count_nonzero(L.diagonal() == 0))
        if model.variant == "lr" and model.zero_local_diagonal_count_:
            warnings.warn(f"{model.zero_local_diagonal_count_} samples are uncovered by the LR neighborhoods; "
                          "the local objective imposes no constraint on these samples. "
                          "Consider explicitly selecting lr_include_self=True.", RuntimeWarning)
        Lg = self._global_laplacian(model.gamma_global)
        scale = max(1.0, model.mu)
        M = L.toarray() / scale + (model.mu / scale) * Lg
        values_all, vectors_all = linalg.eigh(M, subset_by_index=(0, model.n_clusters),
                                             driver="evr", check_finite=False)
        values, F = values_all[:-1], vectors_all[:, :-1]
        model.eigenvalues_ = values * scale
        model.embedding_ = F
        residual = M @ F - F * values
        model.eigen_residual_ = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(M)))
        model.eigenpair_absolute_residuals_ = np.linalg.norm(residual, axis=0) * scale
        model.boundary_eigengap_ = float((values_all[-1] - values[-1]) * scale)
        model.residual_to_gap_ = float(np.linalg.norm(residual) /
            max(values_all[-1] - values[-1], np.finfo(float).tiny))
        model.orthogonality_error_ = float(np.linalg.norm(F.T @ F - np.eye(model.n_clusters)))
        model.graph_objective_ = float(np.sum(F * (L @ F)))
        model.global_objective_ = float(np.sum(F * (Lg @ F)))
        model.spectral_objective_ = model.graph_objective_ + model.mu * model.global_objective_
        self._ridge(model, F, twice=True)
        self._rotate(model, F)
        return model
