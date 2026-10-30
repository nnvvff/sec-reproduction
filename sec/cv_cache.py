"""Fold-local exact reuse for the auditable CV runner.

One FoldCache owns one copied training matrix and one fitted preprocessor.
It never accepts validation/test features or labels. A new fold or a different
preprocessing mode requires a new object. Release the object after its fold to
bound memory; retaining every fold/preprocessing at once is not recommended.

The returned models are the ordinary SEC/baseline classes with their original
predict/save methods. No eigenvalue-limit approximation or restart selection is
introduced. SEC's double-centering SVD is kept separate from the once-centered
baseline SVD, so caching does not change that floating-point convention.
"""

import warnings

import numpy as np
from scipy import linalg, sparse
from sklearn.cluster import KMeans

from .baselines import (DiscriminativeKMeans, LocalLearningClustering,
                        SpectralKMeans, local_learning_operator,
                        resolve_local_rbf_gamma, self_tuning_affinity)
from .model import (_data, SpectralEmbeddedClustering, local_regression_laplacian,
                    sc_laplacian, spectral_rotation)
from .selection import TrainPreprocessor


class FoldCache:
    """Exact intermediates for one fixed training fold and preprocessing mode.

    Usage::

        cache = FoldCache(X_inner_train, preprocessing="standard", fold_id="0")
        model = cache.fit(unfitted_model)
        pred = model.predict(cache.preprocessor.transform(X_validation))

    ``fold_id`` is an audit label only, never a global cache key. The matrix is
    owned by this instance; no global shared cache or content-blind reuse exists.
    ``stats`` reports how many expensive construction operations actually ran.
    """

    def __init__(self, X_train, preprocessing="raw", fold_id=None):
        source = _data(X_train)
        self.preprocessor = TrainPreprocessor(preprocessing).fit(source)
        self.X = np.array(self.preprocessor.transform(source), dtype=float, copy=True, order="C")
        self.X.flags.writeable = False
        self.mean = self.X.mean(axis=0)
        self.mean.flags.writeable = False
        self.n, self.d = self.X.shape
        self.fold_id = fold_id
        self.preprocessing = preprocessing
        self.stats = {"svd_once": 0, "svd_twice": 0, "global_laplacian": 0,
                      "sc_laplacian": 0, "lr_laplacian": 0,
                      "local_learning_operator": 0, "clgr_laplacian": 0}
        self._centered = None
        self._svds = {}
        self._global = {}
        self._sc = {}
        self._lr = {}
        self._local = {}
        self._clgr = {}

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

    def _local_learning(self, model):
        if model.kernel != "rbf" and model.rbf_gamma is not None:
            raise ValueError("rbf_gamma only applies to kernel='rbf'")
        # Requested scale rule belongs in the key, even if resolved values match.
        key = (model.n_neighbors, model.gamma_local, model.kernel, model.rbf_gamma, bool(model.center))
        if key not in self._local:
            Z = self._Z() if model.center else self.X
            resolved = resolve_local_rbf_gamma(Z, model.rbf_gamma) if model.kernel == "rbf" else None
            N = local_learning_operator(Z, model.n_neighbors, model.gamma_local, model.kernel, resolved)
            R = sparse.eye(self.n, format="csr") - N
            self._local[key] = ((R.T @ R).tocsr(), resolved)
            self.stats["local_learning_operator"] += 1
        return self._local[key]

    def _clgr_laplacian(self, model):
        key = (model.n_neighbors, model.scale_neighbor, bool(model.center))
        if key not in self._clgr:
            Z = self._Z() if model.center else self.X
            A = self_tuning_affinity(Z, model.n_neighbors, model.scale_neighbor)
            self._clgr[key] = sparse.diags(np.asarray(A.sum(axis=1)).ravel()) - A
            self.stats["clgr_laplacian"] += 1
        return self._clgr[key]

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

    def fit(self, model):
        """Fit a supported configured model using this fold's fixed training data.

        Parameters of the model determine every matrix cache key. Models remain
        independently usable after the cache is discarded. For KMeans, fitting
        delegates unchanged because its iterations have no shared decomposition.
        """
        if type(model) not in (SpectralEmbeddedClustering, DiscriminativeKMeans,
                              LocalLearningClustering, SpectralKMeans, KMeans):
            raise TypeError("Unsupported model type for exact fold caching")
        if not isinstance(model.n_clusters, (int, np.integer)) or not 1 <= model.n_clusters < self.n:
            raise ValueError("n_clusters must be an integer in [1, n_samples-1]")
        if type(model) is KMeans:
            return model.fit(self.X)
        if type(model) is DiscriminativeKMeans:
            return self._fit_dkm(model)
        self._positive(model.gamma_global, "gamma_global")
        if type(model) is SpectralEmbeddedClustering:
            return self._fit_sec(model)
        if type(model) is LocalLearningClustering:
            return self._fit_ll(model)
        return self._fit_kmr(model)

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

    def _fit_ll(self, model):
        if model.variant not in ("ll", "clgr"):
            raise ValueError("variant must be ll or clgr")
        if not np.isfinite(model.eta) or model.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        Lo, resolved = self._local_learning(model)
        model.kernel_mean_ = self.mean.copy() if model.center else np.zeros(self.d)
        model.rbf_gamma_ = resolved
        model.kernel_config_ = {
            "kernel": model.kernel, "center": bool(model.center),
            "requested_rbf_gamma": model.rbf_gamma, "resolved_rbf_gamma": resolved,
            "fit_samples": self.n,
            "neighborhood_metric": "euclidean before kernel normalization; exclude self",
            "cosine_zero_row_convention": "zero vector; kernel value 0",
            "out_of_sample": "fixed SEC global ridge extension; no kernel refit",
        }
        if model.variant == "clgr":
            scale = max(1.0, model.eta)
            M = (self._clgr_laplacian(model) / scale + (model.eta / scale) * Lo).toarray()
        else:
            scale, M = 1.0, Lo.toarray()
        M = (M + M.T) / 2
        values, F = linalg.eigh(M, subset_by_index=(0, model.n_clusters-1), check_finite=False)
        model.eigenvalues_ = values * scale
        model.spectral_objective_ = float(values.sum() * scale)
        model.objective_ = model.spectral_objective_
        model.eigen_residual_ = float(np.linalg.norm(M @ F - F*values) / max(1.0, np.linalg.norm(M)))
        model.embedding_ = F
        self._ridge(model, F)
        self._rotate(model, F)
        model.orthogonality_error_ = float(np.linalg.norm(F.T @ F - np.eye(model.n_clusters)))
        return model

    def _fit_dkm(self, model):
        self._positive(model.gamma, "gamma")
        if model.prediction_metric not in ("original", "discriminative"):
            raise ValueError("prediction_metric must be original or discriminative")
        model.n_features_in_ = self.d
        model.mean_ = self.mean.copy()
        U, s, Vt = self._svd()
        model.metric_projection_ = Vt.T / np.sqrt(s*s + model.gamma)
        P = U * (s / np.sqrt(s*s + model.gamma))
        km = KMeans(n_clusters=model.n_clusters, init="random", n_init=model.n_init,
                    random_state=model.random_state, max_iter=model.max_iter, tol=model.tol).fit(P)
        model.labels_ = km.labels_
        if len(np.unique(model.labels_)) != model.n_clusters:
            raise ValueError("DKM produced fewer nonempty clusters than requested; check duplicated/degenerate data")
        model.metric_centers_ = km.cluster_centers_
        model.cluster_centers_ = np.array([self.X[model.labels_ == j].mean(axis=0) for j in range(model.n_clusters)])
        model.inertia_ = float(km.inertia_)
        model.objective_ = model.inertia_
        model.trace_objective_ = float(np.sum(P*P) - model.inertia_)
        model.n_iter_ = int(km.n_iter_)
        model.n_clusters_found_ = len(np.unique(model.labels_))
        return model

    def _fit_kmr(self, model):
        Z = self._Z()
        U, s, _ = self._svd()
        count = model.n_clusters - int(model.include_constant)
        rank = np.count_nonzero(s > max(Z.shape) * np.finfo(float).eps * s[0]) if s[0] else 0
        if count > rank:
            raise ValueError("Not enough nonzero principal directions for KM-r; null-space choice is nonunique")
        F, values = U[:, :count].copy(), s[:count] ** 2
        if model.include_constant:
            F = np.column_stack((np.ones(self.n) / np.sqrt(self.n), F))
            values = np.r_[0., values]
        model.eigenvalues_ = values
        residual = Z @ (Z.T @ F) - F * values
        model.eigen_residual_ = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(s*s)))
        model.spectral_objective_ = -float(values.sum())
        model.objective_ = model.spectral_objective_
        model.embedding_ = F
        self._ridge(model, F)
        self._rotate(model, F)
        model.orthogonality_error_ = float(np.linalg.norm(F.T @ F - np.eye(model.n_clusters)))
        return model
