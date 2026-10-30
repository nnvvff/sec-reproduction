"""Orthogonalized, degree-normalized Nyström spectral clustering.

Implements the A/B block construction in Fowlkes et al., TPAMI 2004, Sec. 3,
using the algebraically equivalent low-rank factor of W_hat. It does not build C.
This is a TRANSDUCTIVE cohort method: unseen features affect degrees and the
spectral solution. It is not a fixed inductive predictor like SEC.
"""

import numpy as np
from scipy import linalg

from .model import _data, spectral_rotation


BANDWIDTH_CONVENTIONS = ("inverse-distance", "literal-sigma")


def resolve_rbf_gamma(mean_squared_distance, multiplier=1.0,
                      convention="inverse-distance"):
    """Make the paper's ambiguous sigma convention an explicit choice.

    inverse-distance: gamma = multiplier / mean(d^2), the historical default.
    literal-sigma: sigma = multiplier / mean(d^2), gamma = 1 / sigma^2,
    literally substituting the experiment paragraph into Eq. (1).
    Neither choice authenticates the authors' original implementation.
    """
    if convention not in BANDWIDTH_CONVENTIONS:
        raise ValueError(f"Unknown bandwidth convention: {convention}")
    mean, multiplier = float(mean_squared_distance), float(multiplier)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Mean squared distance must be positive and finite")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("bandwidth_multiplier must be positive and finite")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gamma = (np.float64(multiplier) / mean if convention == "inverse-distance"
                 else np.square(np.float64(mean) / multiplier))
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Resolved RBF gamma is not positive and finite")
    return float(gamma)


def _squared_distances(X, Y=None):
    """Squared distances via GEMM, after a shared shift to reduce cancellation."""
    same = Y is None
    if same:
        Y = X
    origin = X.mean(axis=0)
    X0 = X - origin
    Y0 = X0 if same else Y - origin
    distances = X0 @ Y0.T
    distances *= -2
    distances += np.einsum("ij,ij->i", X0, X0)[:, None]
    distances += np.einsum("ij,ij->i", Y0, Y0)[None, :]
    np.maximum(distances, 0, out=distances)
    if same:
        np.fill_diagonal(distances, 0)
    return distances


def nystrom_embedding_from_blocks(A, B, n_components, rcond=1e-12):
    """Eigenvectors of the degree-normalized Nyström block completion.

    Without numerical truncation W_hat=[[A,B],[B.T,B.T A^-1 B]]. If A is
    truncated, its upper-left block is A_r=Q_r diag(lambda_r) Q_r.T and
    the cross block is Q_r Q_r.T B, not the original A/B.

    A: seen-seen PSD kernel, B: seen-unseen kernel. Near-zero eigenvalues
    of A are truncated at rcond*lambda_max; this numerical choice is reported.
    Positive A eigenvalues give factor T=[Q sqrt(lambda); B.T Q/sqrt(lambda)].
    Orthogonalize normalized T via its Gram matrix, exactly equivalent to the
    single-shot orthogonalized Nyström eigensystem for nonsingular A.
    """
    A, B = _data(A), _data(B)
    if A.shape[0] != A.shape[1] or B.shape[0] != A.shape[0]:
        raise ValueError("A must be square and B must have the same number of rows")
    if not np.allclose(A, A.T, rtol=1e-10, atol=1e-12):
        raise ValueError("A must be symmetric")
    if not 0 < rcond < 1:
        raise ValueError("rcond must be in (0,1)")
    eig, Q = linalg.eigh(A, check_finite=False)
    cutoff = max(float(eig[-1]), np.finfo(float).eps) * rcond
    if eig[0] < -cutoff:
        raise ValueError("Nyström orthogonalization requires PSD A")
    keep = eig > cutoff
    if np.count_nonzero(keep) < n_components:
        raise ValueError("Retained Nyström rank is smaller than n_components")
    positive, Q = eig[keep], Q[:, keep]
    factor = np.vstack((Q * np.sqrt(positive), (B.T @ Q) / np.sqrt(positive)))
    degrees = factor @ factor.sum(axis=0)
    if np.any(degrees <= 0) or not np.isfinite(degrees).all():
        raise ValueError("Nyström approximation has nonpositive/nonfinite degrees")
    normalized = factor / np.sqrt(degrees[:, None])
    gram = normalized.T @ normalized
    values, V = linalg.eigh(gram, subset_by_index=(len(positive)-n_components, len(positive)-1),
                            check_finite=False)
    values, V = values[::-1], V[:, ::-1]
    if np.any(values <= 0):
        raise ValueError("Requested embedding has nonpositive eigenvalues")
    F = (normalized @ V) / np.sqrt(values)
    residual = normalized @ (normalized.T @ F) - F * values
    diagnostics = {"retained_rank": int(keep.sum()), "sample_size": len(A),
                   "truncated_eigenvalues": int((~keep).sum()), "rcond": rcond,
                   "min_approx_degree": float(degrees.min()),
                   "eigen_residual": float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(gram))),
                   "orthogonality_error": float(np.linalg.norm(F.T @ F - np.eye(n_components)))}
    return F, values, diagnostics


class NystromSpectralClustering:
    """RBF kernel Nyström with spectral rotation of the entire cohort.

    By default gamma = bandwidth_multiplier / mean squared seen-seen distance.
    This explicitly treats the SEC paper's reciprocal distance parameter as an
    inverse RBF bandwidth. Its text is inconsistent with Eq. (1)'s sigma units;
    report this convention and do not claim exact author baseline parity.
    ``bandwidth_convention='literal-sigma'`` instead directly substitutes the
    experiment paragraph's sigma into Eq. (1). No labels enter this model;
    any oracle selection is performed and labelled separately by the runner.
    Diagonal kernel entries are 1.
    """

    def __init__(self, n_clusters, rbf_gamma=None, bandwidth_multiplier=1.0,
                 n_init=50, random_state=0, rcond=1e-12, max_iter=300,
                 bandwidth_convention="inverse-distance"):
        self.n_clusters = n_clusters
        self.rbf_gamma = rbf_gamma
        self.bandwidth_multiplier = bandwidth_multiplier
        self.n_init = n_init
        self.random_state = random_state
        self.rcond = rcond
        self.max_iter = max_iter
        self.bandwidth_convention = bandwidth_convention

    def fit(self, X_seen, X_unseen):
        X_seen, X_unseen = _data(X_seen), _data(X_unseen)
        if X_seen.shape[1] != X_unseen.shape[1]:
            raise ValueError("Seen and unseen feature dimensions differ")
        if not isinstance(self.n_clusters, (int, np.integer)) or not 1 <= self.n_clusters < len(X_seen):
            raise ValueError("n_clusters must be an integer smaller than seen sample size")
        if not np.isfinite(self.bandwidth_multiplier) or self.bandwidth_multiplier <= 0:
            raise ValueError("bandwidth_multiplier must be positive and finite")
        if self.bandwidth_convention not in BANDWIDTH_CONVENTIONS:
            raise ValueError(f"Unknown bandwidth convention: {self.bandwidth_convention}")
        seen_distances = _squared_distances(X_seen)
        self.mean_squared_seen_distance_ = float(seen_distances.mean())
        if self.rbf_gamma is None:
            self.rbf_gamma_ = resolve_rbf_gamma(self.mean_squared_seen_distance_,
                                              self.bandwidth_multiplier,
                                              self.bandwidth_convention)
        else:
            self.rbf_gamma_ = float(self.rbf_gamma)
            if not np.isfinite(self.rbf_gamma_) or self.rbf_gamma_ <= 0:
                raise ValueError("rbf_gamma must be positive and finite")
        A = seen_distances
        A *= -self.rbf_gamma_
        np.exp(A, out=A)
        B = _squared_distances(X_seen, X_unseen)
        B *= -self.rbf_gamma_
        np.exp(B, out=B)
        # Expose kernel collapse for literal-sigma at raw pixel scales.
        self.kernel_zero_fraction_ = float(np.count_nonzero(B == 0) / B.size)
        self.embedding_, self.eigenvalues_, self.diagnostics_ = nystrom_embedding_from_blocks(
            A, B, self.n_clusters, self.rcond)
        rotated = spectral_rotation(self.embedding_, self.n_init, self.random_state, self.max_iter)
        self.rotation_ = rotated.rotation
        self.labels_ = rotated.labels
        self.seen_labels_ = self.labels_[:len(X_seen)]
        self.unseen_labels_ = self.labels_[len(X_seen):]
        self.rotation_objective_ = rotated.objective
        self.rotation_converged_ = rotated.converged
        self.eigen_residual_ = self.diagnostics_["eigen_residual"]
        self.orthogonality_error_ = self.diagnostics_["orthogonality_error"]
        self.n_clusters_found_ = len(np.unique(self.labels_))
        self.spectral_objective_ = float(self.n_clusters - self.eigenvalues_.sum())
        return self

    def fit_predict(self, X_seen, X_unseen):
        self.fit(X_seen, X_unseen)
        return self.seen_labels_, self.unseen_labels_

    def fit_predict_seen_unseen(self, X_seen, X_unseen):
        """Explicit cohort API: both feature sets enter the transductive solve."""
        return self.fit_predict(X_seen, X_unseen)

    def save(self, path):
        if not hasattr(self, "labels_"):
            raise RuntimeError("Fit before saving")
        # Cohort solution only: intentionally no misleading single-point predictor.
        np.savez_compressed(path, seen_labels=self.seen_labels_, unseen_labels=self.unseen_labels_,
                            embedding=self.embedding_, eigenvalues=self.eigenvalues_,
                            rotation=self.rotation_, rbf_gamma=self.rbf_gamma_, rcond=self.rcond,
                            mean_squared_seen_distance=self.mean_squared_seen_distance_,
                            bandwidth_convention=("explicit-gamma" if self.rbf_gamma is not None
                                                  else self.bandwidth_convention),
                            bandwidth_multiplier=self.bandwidth_multiplier)
