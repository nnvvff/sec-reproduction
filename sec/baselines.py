"""Formula-based baselines for Nie et al. TNN 2011, with explicit ambiguities.

Samples are rows. These are independent implementations, not author code.
LL/CLGR expose the local kernel because the SEC paper does not specify it
completely. Default names in reports should be LL-linear / CLGR-linear.
"""

import json
import warnings

import numpy as np
from scipy import linalg, sparse
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from .model import _data, neighbor_indices, normalize_rows, spectral_rotation


def _validate(X, n_clusters, gamma):
    X = _data(X)
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters < len(X):
        raise ValueError("n_clusters must be an integer in [1,n_samples-1]")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("regularization must be positive and finite")
    return X


def _ridge_from_embedding(X, F, gamma):
    mean = X.mean(axis=0)
    U, s, Vt = linalg.svd(X - mean, full_matrices=False, check_finite=False)
    coef = (Vt.T * (s / (s*s + gamma))) @ (U.T @ F)
    return mean, coef, F.mean(axis=0)


class _SpectralRidgeModel:
    def _finish(self, X, F):
        self.embedding_ = F
        self.n_features_in_ = X.shape[1]
        self.mean_, self.coef_, self.embedding_bias_ = _ridge_from_embedding(X, F, self.gamma_global)
        self.intercept_ = self.embedding_bias_ - self.mean_ @ self.coef_
        rotation = spectral_rotation(F, self.n_init, self.random_state, self.max_iter)
        self.labels_ = rotation.labels
        self.rotation_ = rotation.rotation
        self.rotation_objective_ = rotation.objective
        self.rotation_history_ = rotation.history
        self.rotation_converged_ = rotation.converged
        self.n_clusters_found_ = len(np.unique(self.labels_))
        self.orthogonality_error_ = float(np.linalg.norm(F.T @ F - np.eye(self.n_clusters)))
        return self

    def transform(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit the model before prediction")
        X = _data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from training")
        return (X - self.mean_) @ self.coef_ + self.embedding_bias_

    def predict(self, X):
        return (normalize_rows(self.transform(X)) @ self.rotation_).argmax(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def save(self, path):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit before saving")
        np.savez_compressed(path, mean=self.mean_, coef=self.coef_,
                            embedding_bias=self.embedding_bias_, rotation=self.rotation_,
                            labels=self.labels_, embedding=self.embedding_,
                            eigenvalues=self.eigenvalues_)


class SpectralKMeans(_SpectralRidgeModel):
    """KM-r: top c eigenvectors of centered X X.T, Eq. (27), spectral rotation.

    The paper's Eqs. (29)-(30) leave an ambiguity about the constant vector.
    Default uses Eq. (27) literally. include_constant=True instead uses
    constant + (c-1) principal directions (the SEC mu->infinity convention),
    and MUST be labeled as a separate variant in experiments.
    """

    def __init__(self, n_clusters, gamma_global=1.0, n_init=50,
                 random_state=0, include_constant=False, max_iter=300):
        self.n_clusters = n_clusters
        self.gamma_global = gamma_global
        self.n_init = n_init
        self.random_state = random_state
        self.include_constant = include_constant
        self.max_iter = max_iter

    def fit(self, X):
        X = _validate(X, self.n_clusters, self.gamma_global)
        Z = X - X.mean(axis=0)
        U, s, _ = linalg.svd(Z, full_matrices=False, check_finite=False)
        count = self.n_clusters - int(self.include_constant)
        rank = np.count_nonzero(s > max(Z.shape) * np.finfo(float).eps * s[0]) if s[0] else 0
        if count > rank:
            raise ValueError("Not enough nonzero principal directions for KM-r; null-space choice is nonunique")
        F = U[:, :count]
        values = s[:count] ** 2
        if self.include_constant:
            F = np.column_stack((np.ones(len(X)) / np.sqrt(len(X)), F))
            values = np.r_[0., values]
        self.eigenvalues_ = values
        residual = Z @ (Z.T @ F) - F * values
        self.eigen_residual_ = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(s*s)))
        self.spectral_objective_ = -float(values.sum())
        self.objective_ = self.spectral_objective_
        return self._finish(X, F)


class DiscriminativeKMeans:
    """Exact factorization of the Eq. (34) kernel followed by Lloyd K-means.

    P=U diag(s/sqrt(s^2+gamma)); P P.T = Z(Z.T Z+gamma I)^-1 Z.T.
    Minimizing K-means inertia on P maximizes the discrete Eq. (34) trace.
    prediction_metric='original' uses nearest original-space cluster centers,
    following Section VI's literal description. 'discriminative' uses the
    learned metric; this unresolved OOS convention must be reported explicitly.
    """

    def __init__(self, n_clusters, gamma=1.0, n_init=50, random_state=0,
                 prediction_metric="original", max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.n_init = n_init
        self.random_state = random_state
        self.prediction_metric = prediction_metric
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        X = _validate(X, self.n_clusters, self.gamma)
        if self.prediction_metric not in ("original", "discriminative"):
            raise ValueError("prediction_metric must be original or discriminative")
        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0)
        U, s, Vt = linalg.svd(X - self.mean_, full_matrices=False, check_finite=False)
        # Small singular directions contribute negligible training kernel values;
        # retaining them is harmless and preserves the exact factorization.
        self.metric_projection_ = Vt.T / np.sqrt(s*s + self.gamma)
        P = U * (s / np.sqrt(s*s + self.gamma))
        km = KMeans(n_clusters=self.n_clusters, init="random", n_init=self.n_init,
                    random_state=self.random_state, max_iter=self.max_iter, tol=self.tol).fit(P)
        self.labels_ = km.labels_
        if len(np.unique(self.labels_)) != self.n_clusters:
            raise ValueError("DKM produced fewer nonempty clusters than requested; check duplicated/degenerate data")
        self.metric_centers_ = km.cluster_centers_
        self.cluster_centers_ = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.n_clusters)])
        self.inertia_ = float(km.inertia_)
        self.objective_ = self.inertia_
        self.trace_objective_ = float(np.sum(P*P) - self.inertia_)
        self.n_iter_ = int(km.n_iter_)
        self.n_clusters_found_ = len(np.unique(self.labels_))
        return self

    def transform(self, X):
        if not hasattr(self, "metric_projection_"):
            raise RuntimeError("Fit before prediction")
        X = _data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from training")
        return (X - self.mean_) @ self.metric_projection_

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Fit before prediction")
        X = _data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from training")
        if self.prediction_metric == "original":
            return pairwise_distances_argmin(X, self.cluster_centers_)
        return pairwise_distances_argmin(self.transform(X), self.metric_centers_)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def save(self, path):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Fit before saving")
        np.savez_compressed(path, mean=self.mean_, projection=self.metric_projection_,
                            original_centers=self.cluster_centers_, metric_centers=self.metric_centers_,
                            prediction_metric=self.prediction_metric, labels=self.labels_, gamma=self.gamma)


def self_tuning_affinity(X, k=5, scale_neighbor=7):
    """Union kNN Gaussian weights, same convention as sec.model.sc_laplacian."""
    X = _data(X)
    rank = min(scale_neighbor, len(X) - 1)
    distances, indices = neighbor_indices(X, max(k, rank))
    scales = distances[:, rank - 1]
    positive = distances[distances > 0]
    floor = max(float(positive.min()) * 1e-12, np.finfo(float).eps) if positive.size else 1.0
    scales = np.maximum(scales, floor)
    rows, cols = np.repeat(np.arange(len(X)), k), indices[:, :k].ravel()
    weights = np.exp(-distances[:, :k].ravel() ** 2 / (scales[rows] * scales[cols]))
    A = sparse.csr_matrix((weights, (rows, cols)), shape=(len(X), len(X)))
    return A.maximum(A.T)


def resolve_local_rbf_gamma(X, rbf_gamma):
    """Resolve an explicitly requested RBF rule using only the supplied fit data.

    All returned values multiply squared distance in exp(-gamma*d^2).
    ``mean-distance`` averages all n^2 pairwise distances, including diagonal
    zeros. ``ll2006-mean-norm`` is the central Gaussian width from the original
    LL paper Section 4.3: width=(mean ||x_i||)^2, hence gamma=1/width.
    Neither rule is asserted to be the missing SEC-2011 experimental setting.
    """
    X = _data(X)
    if isinstance(rbf_gamma, str):
        if rbf_gamma == "mean-distance":
            Z = X - X.mean(axis=0)
            denominator = 2 * np.mean(np.sum(Z * Z, axis=1))
        elif rbf_gamma == "ll2006-mean-norm":
            denominator = np.mean(np.linalg.norm(X, axis=1)) ** 2
        else:
            raise ValueError("rbf_gamma must be positive, mean-distance or ll2006-mean-norm")
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Requested RBF scale is undefined on zero-scale fit data")
        gamma = 1.0 / denominator
    else:
        if rbf_gamma is None or isinstance(rbf_gamma, (bool, np.bool_)):
            raise ValueError("RBF local learning requires an explicit positive rbf_gamma or scale rule")
        try:
            gamma = float(rbf_gamma)
        except (TypeError, ValueError) as error:
            raise ValueError("rbf_gamma must be positive, mean-distance or ll2006-mean-norm") from error
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Resolved RBF gamma must be finite and positive")
    return float(gamma)


def local_learning_operator(X, k=5, gamma=1.0, kernel="linear", rbf_gamma=None):
    """Neighbor KRR predictor N; Lo=(I-N).T(I-N), SEC Eqs. (24)-(25).

    Unlike SEC/LR, each target xi is excluded from its training neighborhood.
    Kernel variants follow Wu & Scholkopf NIPS 2006 Eq. (6)-(10).
    """
    X = _data(X)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if kernel not in ("linear", "cosine", "rbf"):
        raise ValueError("kernel must be linear, cosine or rbf")
    if kernel == "rbf":
        rbf_gamma = resolve_local_rbf_gamma(X, rbf_gamma)
    elif rbf_gamma is not None:
        raise ValueError("rbf_gamma only applies to kernel='rbf'")
    _, neighbors = neighbor_indices(X, k)
    coefficients = np.empty((len(X), k))
    features = normalize_rows(X) if kernel == "cosine" else X
    for i, ids in enumerate(neighbors):
        Xi, xi = features[ids], features[i]
        if kernel == "rbf":
            sq = np.sum((Xi[:, None, :] - Xi[None, :, :])**2, axis=2)
            K = np.exp(-rbf_gamma * sq)
            target = np.exp(-rbf_gamma * np.sum((Xi - xi)**2, axis=1))
        else:
            K, target = Xi @ Xi.T, Xi @ xi
        coefficients[i] = linalg.solve(K + gamma*np.eye(k), target, assume_a="pos")
    rows = np.repeat(np.arange(len(X)), k)
    return sparse.csr_matrix((coefficients.ravel(), (rows, neighbors.ravel())), shape=(len(X), len(X)))


class LocalLearningClustering(_SpectralRidgeModel):
    """LL kernel KRR residual or CLGR graph + local KRR residual.

    LL: min tr(F.T Lo F). CLGR: min tr(F.T (L+eta Lo) F), Eq. (23).
    L is UNNORMALIZED, as in Eq. (23); SC's Ltilde must not replace it.
    predict() is SEC's gamma_global ridge extension at mu=0, not local KRR.
    Kernel preprocessing and any requested RBF scale use fit data only.
    With cosine, zero feature rows map to the zero vector (kernel value 0).
    Neighborhoods always use Euclidean distance before kernel row normalization.
    """

    def __init__(self, n_clusters, variant="ll", gamma_local=1.0, eta=1.0,
                 gamma_global=1.0, n_neighbors=5, scale_neighbor=7, n_init=50,
                 random_state=0, kernel="linear", rbf_gamma=None, max_iter=300,
                 center=True):
        self.n_clusters = n_clusters
        self.variant = variant
        self.gamma_local = gamma_local
        self.eta = eta
        self.gamma_global = gamma_global
        self.n_neighbors = n_neighbors
        self.scale_neighbor = scale_neighbor
        self.n_init = n_init
        self.random_state = random_state
        self.kernel = kernel
        self.rbf_gamma = rbf_gamma
        self.max_iter = max_iter
        self.center = center

    def fit(self, X):
        X = _validate(X, self.n_clusters, self.gamma_global)
        if self.variant not in ("ll", "clgr"):
            raise ValueError("variant must be ll or clgr")
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        Z = X - X.mean(axis=0) if self.center else X
        self.kernel_mean_ = X.mean(axis=0) if self.center else np.zeros(X.shape[1])
        self.rbf_gamma_ = resolve_local_rbf_gamma(Z, self.rbf_gamma) if self.kernel == "rbf" else None
        if self.kernel != "rbf" and self.rbf_gamma is not None:
            raise ValueError("rbf_gamma only applies to kernel='rbf'")
        self.kernel_config_ = {
            "kernel": self.kernel,
            "center": bool(self.center),
            "requested_rbf_gamma": self.rbf_gamma,
            "resolved_rbf_gamma": self.rbf_gamma_,
            "fit_samples": len(X),
            "neighborhood_metric": "euclidean before kernel normalization; exclude self",
            "cosine_zero_row_convention": "zero vector; kernel value 0",
            "out_of_sample": "fixed SEC global ridge extension; no kernel refit",
        }
        N = local_learning_operator(Z, self.n_neighbors, self.gamma_local, self.kernel, self.rbf_gamma_)
        residual = sparse.eye(len(X), format="csr") - N
        Lo = residual.T @ residual
        if self.variant == "clgr":
            A = self_tuning_affinity(Z, self.n_neighbors, self.scale_neighbor)
            L = sparse.diags(np.asarray(A.sum(axis=1)).ravel()) - A
            scale = max(1.0, self.eta)
            M = (L / scale + (self.eta / scale) * Lo).toarray()
        else:
            scale = 1.0
            M = Lo.toarray()
        M = (M + M.T) / 2
        values, F = linalg.eigh(M, subset_by_index=(0, self.n_clusters-1), check_finite=False)
        self.eigenvalues_ = values * scale
        self.spectral_objective_ = float(values.sum() * scale)
        self.objective_ = self.spectral_objective_
        self.eigen_residual_ = float(np.linalg.norm(M @ F - F*values) / max(1.0, np.linalg.norm(M)))
        return self._finish(X, F)

    def save(self, path):
        """Retain historical prediction-array keys and add explicit fit settings."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("Fit before saving")
        if self.kernel_config_.get("source") == "legacy_unrecorded":
            # Do not retrofit inferred defaults into a historical model.
            return super().save(path)
        params = {name: getattr(self, name) for name in (
            "n_clusters", "variant", "gamma_local", "eta", "gamma_global",
            "n_neighbors", "scale_neighbor", "n_init", "random_state", "kernel",
            "rbf_gamma", "max_iter", "center")}
        # np scalar constructor arguments are valid; serialize them as scalars.
        scalar = lambda value: value.item() if isinstance(value, np.generic) else value
        metadata = {"schema": 1, "parameters": params, "kernel_config": self.kernel_config_}
        np.savez_compressed(path, mean=self.mean_, coef=self.coef_,
                            embedding_bias=self.embedding_bias_, rotation=self.rotation_,
                            labels=self.labels_, embedding=self.embedding_,
                            eigenvalues=self.eigenvalues_, kernel_mean=self.kernel_mean_,
                            configuration_json=json.dumps(metadata, default=scalar, sort_keys=True))

    @classmethod
    def load(cls, path):
        """Load the fixed ridge predictor, including legacy prediction-only NPZs.

        Legacy files contain no kernel metadata. They retain prediction behavior
        but expose ``kernel_config_['source']='legacy_unrecorded'`` and warn;
        defaults must not be interpreted as evidence of the historical fit.
        """
        with np.load(path, allow_pickle=False) as state:
            if "configuration_json" in state:
                metadata = json.loads(str(state["configuration_json"]))
                if metadata.get("schema") != 1:
                    raise ValueError("Unsupported local-learning model schema")
                model = cls(**metadata["parameters"])
                model.kernel_config_ = metadata["kernel_config"]
                model.rbf_gamma_ = model.kernel_config_["resolved_rbf_gamma"]
                model.kernel_mean_ = state["kernel_mean"].copy()
            else:
                warnings.warn("Legacy model has no recorded LL kernel configuration; "
                              "loaded prediction arrays only", UserWarning)
                model = cls(state["rotation"].shape[0])
                model.kernel_config_ = {"source": "legacy_unrecorded"}
                model.rbf_gamma_ = None
                model.kernel_mean_ = None
            for attribute, key in (
                ("mean_", "mean"), ("coef_", "coef"),
                ("embedding_bias_", "embedding_bias"), ("rotation_", "rotation"),
                ("labels_", "labels"), ("embedding_", "embedding"),
                ("eigenvalues_", "eigenvalues")):
                setattr(model, attribute, state[key].copy())
        model.n_features_in_ = model.coef_.shape[0]
        model.intercept_ = model.embedding_bias_ - model.mean_ @ model.coef_
        model.n_clusters_found_ = len(np.unique(model.labels_))
        return model
