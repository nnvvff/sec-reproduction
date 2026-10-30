"""SEC/SC and SEC/LR, with samples stored in ROWS (transpose of paper X).

Equations refer to the supplied 2011 TNN paper, not its 2009 predecessor.
Local centering in Eq. (20) is retained when eliminating Wi in Eq. (21).
"""

from dataclasses import dataclass
import json
import warnings

import numpy as np
from scipy import linalg, sparse
from sklearn.neighbors import NearestNeighbors


def _data(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite (n_samples, n_features) matrix")
    return X


def normalize_rows(F):
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    return F / np.maximum(norms, np.finfo(float).eps)


def neighbor_indices(X, k):
    """k distinct other samples; explicitly exclude self even with duplicates."""
    X = _data(X)
    if not isinstance(k, (int, np.integer)) or not 1 <= k < len(X):
        raise ValueError("n_neighbors must be an integer in [1, n_samples - 1]")
    distances, indices = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    out_i = np.empty((len(X), k), dtype=int)
    out_d = np.empty((len(X), k))
    for row in range(len(X)):
        keep = indices[row] != row
        out_i[row] = indices[row, keep][:k]
        out_d[row] = distances[row, keep][:k]
    return out_d, out_i


def sc_laplacian(X, k=5, scale_neighbor=7):
    """Union k-NN graph; self-tuning exp(-dij^2/(sigma_i sigma_j)).

    sigma_i is distance to the 7th OTHER neighbor by default (self-tuning
    SC convention). The SEC paper specifies graph k=5 but not scale rank;
    expose the latter explicitly rather than conflating the two choices.
    """
    X = _data(X)
    if not isinstance(scale_neighbor, (int, np.integer)) or scale_neighbor < 1:
        raise ValueError("scale_neighbor must be a positive integer")
    rank = min(scale_neighbor, len(X) - 1)
    distances, indices = neighbor_indices(X, max(k, rank))
    scales = distances[:, rank - 1]
    positive = distances[distances > 0]
    floor = max(float(positive.min()) * 1e-12, np.finfo(float).eps) if positive.size else 1.0
    scales = np.maximum(scales, floor)
    cols = indices[:, :k].ravel()
    rows = np.repeat(np.arange(len(X)), k)
    weights = np.exp(-distances[:, :k].ravel() ** 2 / (scales[rows] * scales[cols]))
    A = sparse.csr_matrix((weights, (rows, cols)), shape=(len(X), len(X)))
    A = A.maximum(A.T)
    degrees = np.asarray(A.sum(axis=1)).ravel()
    if np.any(degrees <= 0):
        raise ValueError("Graph has zero-degree vertices; check data or neighbor settings")
    D = sparse.diags(1 / np.sqrt(degrees))
    return (sparse.eye(len(X), format="csr") - D @ A @ D).tocsr()


def local_regression_laplacian(X, k=5, gamma=1.0, include_self=True):
    """Eqs. (19)-(21): sum S_i H_i gamma(K_i + gamma I)^-1 H_i S_i^T.

    K_i = (H_i X_i)(H_i X_i)^T with ROW samples. Solve k x k systems,
    never d x d inverses. Neighborhood has exactly k points. By default
    these are center + (k-1) others, matching the benchmark convention.
    include_self=False explicitly selects the historical exclusion ablation.
    The paper does not resolve this convention.
    """
    X = _data(X)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if include_self:
        if not isinstance(k, (int, np.integer)) or not 2 <= k <= len(X):
            raise ValueError("Self-inclusive local neighborhoods require 2 <= k <= n")
        _, others = neighbor_indices(X, k - 1)
        neighbors = np.column_stack((np.arange(len(X)), others))
    else:
        _, neighbors = neighbor_indices(X, k)
    H = np.eye(k) - np.ones((k, k)) / k
    blocks = np.empty((len(X), k, k))
    for i, ids in enumerate(neighbors):
        Z = X[ids] - X[ids].mean(axis=0)
        block = gamma * H @ linalg.solve(Z @ Z.T + gamma * np.eye(k), H, assume_a="pos")
        blocks[i] = (block + block.T) / 2
    rows = np.repeat(neighbors, k, axis=1).ravel()
    cols = np.tile(neighbors, (1, k)).ravel()
    return sparse.coo_matrix((blocks.ravel(), (rows, cols)), shape=(len(X), len(X))).tocsr()


def global_regression_laplacian(X, gamma=1.0):
    """Eq. (17), using thin SVD; returns Lg and the SVD for ridge prediction."""
    X = _data(X)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    Z = X - X.mean(axis=0)
    U, s, Vt = linalg.svd(Z, full_matrices=False, check_finite=False)
    shrink = s**2 / (s**2 + gamma)
    Lg = np.eye(len(X)) - np.full((len(X), len(X)), 1 / len(X))
    Lg -= (U * shrink) @ U.T
    return (Lg + Lg.T) / 2, (U, s, Vt)


@dataclass
class RotationResult:
    labels: np.ndarray
    rotation: np.ndarray
    objective: float
    history: list
    converged: bool


def spectral_rotation(F, n_init=50, random_state=0, max_iter=300, tol=1e-10):
    """Row normalization (6), alternating one-hot / Procrustes optimization (7).

    Retain ALL c eigenvectors, including the smallest eigenvalue's vector.
    Choose restarts by Eq. (7)'s objective, never by ground-truth accuracy.
    """
    F = _data(F)
    n, c = F.shape
    if c > n or n_init < 1 or max_iter < 1:
        raise ValueError("Require c <= n, n_init >= 1, max_iter >= 1")
    Z = normalize_rows(F)
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(n_init):
        # Yu-Shi style initialization: random first row, near-orthogonal others.
        R = np.empty((c, c))
        chosen = [int(rng.integers(n))]
        R[:, 0] = Z[chosen[0]]
        similarity = np.zeros(n)
        for j in range(1, c):
            similarity += np.abs(Z @ R[:, j - 1])
            candidates = similarity.copy()
            candidates[chosen] = np.inf
            chosen.append(int(np.argmin(candidates)))
            R[:, j] = Z[chosen[-1]]
        history = []
        previous_labels = None
        converged = False
        for _iteration in range(max_iter):
            labels = np.argmax(Z @ R, axis=1)
            cross = np.zeros((c, c))
            np.add.at(cross.T, labels, Z)
            u, _, vt = linalg.svd(cross, check_finite=False)
            R = u @ vt
            rotated = Z @ R
            objective = float(np.sum(Z * Z) + n - 2 * rotated[np.arange(n), labels].sum())
            history.append(objective)
            updated = np.argmax(rotated, axis=1)
            if np.array_equal(updated, labels):
                converged = True
                break
            if previous_labels is not None and abs(history[-2] - objective) <= tol:
                # Still return labels consistent with the returned rotation.
                break
            previous_labels = labels
        labels = np.argmax(Z @ R, axis=1)
        objective = float(np.sum(Z * Z) + n - 2 * (Z @ R)[np.arange(n), labels].sum())
        result = RotationResult(labels, R, objective, history, converged)
        if best is None or result.objective < best.objective:
            best = result
    if not best.converged:
        warnings.warn("Best spectral rotation did not reach a fixed point", RuntimeWarning)
    return best


class SpectralEmbeddedClustering:
    """Dense exact spectral solve of SEC, intended for a few thousand seen points.

    fit(X) accepts no labels. labels_ discretizes the optimized embedding;
    predict(X) uses its ridge approximation and may differ on training points.
    For mu=0, labels_ is SC/LR and predict is the paper's SEC extension at mu->0.
    """

    def __init__(self, n_clusters, variant="sc", mu=1.0, gamma_global=1.0,
                 gamma_local=1.0, n_neighbors=5, scale_neighbor=7,
                 n_init=50, random_state=0, max_iter=300, lr_include_self=True):
        self.n_clusters = n_clusters
        self.variant = variant
        self.mu = mu
        self.gamma_global = gamma_global
        self.gamma_local = gamma_local
        self.n_neighbors = n_neighbors
        self.scale_neighbor = scale_neighbor
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter
        self.lr_include_self = lr_include_self

    def fit(self, X):
        X = _data(X)
        n, d = X.shape
        if not isinstance(self.n_clusters, (int, np.integer)) or not 1 <= self.n_clusters < n:
            raise ValueError("n_clusters must be an integer in [1, n_samples - 1]")
        if self.variant not in ("sc", "lr"):
            raise ValueError("variant must be 'sc' or 'lr'")
        if not np.isfinite(self.mu) or self.mu < 0:
            raise ValueError("mu must be finite and nonnegative")
        self.mean_ = X.mean(axis=0)
        self.n_features_in_ = d
        Z = X - self.mean_
        L = (sc_laplacian(Z, self.n_neighbors, self.scale_neighbor) if self.variant == "sc"
             else local_regression_laplacian(Z, self.n_neighbors, self.gamma_local, self.lr_include_self))
        self.zero_local_diagonal_count_ = int(np.count_nonzero(L.diagonal() == 0))
        if self.variant == "lr" and self.zero_local_diagonal_count_:
            warnings.warn(f"{self.zero_local_diagonal_count_} samples are uncovered by the LR neighborhoods; "
                          "the local objective imposes no constraint on these samples. "
                          "Consider explicitly selecting lr_include_self=True.", RuntimeWarning)
        Lg, (U, s, Vt) = global_regression_laplacian(Z, self.gamma_global)
        # Positive rescaling changes eigenvalues only; avoids overflow for large mu.
        scale = max(1.0, self.mu)
        M = L.toarray() / scale + (self.mu / scale) * Lg
        values_all, vectors_all = linalg.eigh(M, subset_by_index=(0, self.n_clusters),
                               driver="evr", check_finite=False)
        values, F = values_all[:-1], vectors_all[:, :-1]
        self.eigenvalues_ = values * scale
        self.embedding_ = F
        residual_vectors = M @ F - F * values
        self.eigen_residual_ = float(np.linalg.norm(residual_vectors) / max(1.0, np.linalg.norm(M)))
        self.eigenpair_absolute_residuals_ = np.linalg.norm(residual_vectors, axis=0) * scale
        self.boundary_eigengap_ = float((values_all[-1] - values[-1]) * scale)
        self.residual_to_gap_ = float(np.linalg.norm(residual_vectors) /
                                     max(values_all[-1] - values[-1], np.finfo(float).tiny))
        self.orthogonality_error_ = float(np.linalg.norm(F.T @ F - np.eye(self.n_clusters)))
        self.graph_objective_ = float(np.sum(F * (L @ F)))
        self.global_objective_ = float(np.sum(F * (Lg @ F)))
        self.spectral_objective_ = self.graph_objective_ + self.mu * self.global_objective_
        # Eq. (14), stable SVD form, with train-only centering.
        self.coef_ = (Vt.T * (s / (s**2 + self.gamma_global))) @ (U.T @ F)
        self.embedding_bias_ = F.mean(axis=0)
        self.intercept_ = self.embedding_bias_ - self.mean_ @ self.coef_
        result = spectral_rotation(F, self.n_init, self.random_state, self.max_iter)
        self.labels_ = result.labels
        self.rotation_ = result.rotation
        self.rotation_objective_ = result.objective
        self.rotation_history_ = result.history
        self.rotation_converged_ = result.converged
        self.n_clusters_found_ = len(np.unique(self.labels_))
        return self

    def transform(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before transform or predict")
        X = _data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from training data")
        return (X - self.mean_) @ self.coef_ + self.embedding_bias_

    def predict(self, X):
        return np.argmax(normalize_rows(self.transform(X)) @ self.rotation_, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def save(self, path):
        """Portable inference state, no pickle; (X-mean) @ coef + bias then rotation."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before save")
        np.savez_compressed(path, mean=self.mean_, coef=self.coef_,
                            embedding_bias=self.embedding_bias_, rotation=self.rotation_,
                            labels=self.labels_, embedding=self.embedding_,
                            eigenvalues=self.eigenvalues_,
                            parameters_json=json.dumps({k: getattr(self, k) for k in (
                                "n_clusters", "variant", "mu", "gamma_global", "gamma_local",
                                "n_neighbors", "scale_neighbor", "n_init", "random_state",
                                "max_iter", "lr_include_self")}))

    @classmethod
    def load(cls, path):
        """Load the portable inference state without pickle or training data."""
        with np.load(path, allow_pickle=False) as state:
            params = json.loads(str(state["parameters_json"])) if "parameters_json" in state else {
                "n_clusters": state["coef"].shape[1], "lr_include_self": False}
            model = cls(**params)
            for name, key in (("mean_", "mean"), ("coef_", "coef"), ("embedding_bias_", "embedding_bias"),
                              ("rotation_", "rotation"), ("labels_", "labels"),
                              ("embedding_", "embedding"), ("eigenvalues_", "eigenvalues")):
                setattr(model, name, state[key].copy())
        model.n_features_in_ = model.coef_.shape[0]
        model.intercept_ = model.embedding_bias_ - model.mean_ @ model.coef_
        return model
