import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg
from sklearn.datasets import make_blobs

from sec.model import (SpectralEmbeddedClustering, global_regression_laplacian,
                       local_regression_laplacian, neighbor_indices,
                       normalize_rows, sc_laplacian, spectral_rotation)
from sec.metrics import clustering_accuracy, fit_label_mapping, mapped_accuracy


@pytest.mark.parametrize("shape", [(31, 7), (12, 24)])
def test_global_elimination_equals_original_ridge_objective(shape):
    rng = np.random.default_rng(42)
    X = rng.normal(size=shape) + 3
    Z = X - X.mean(axis=0)
    F = linalg.qr(rng.normal(size=(shape[0], 3)), mode="economic")[0]
    gamma = 1.7
    Lg, _ = global_regression_laplacian(X, gamma)
    W = linalg.solve(Z.T @ Z + gamma * np.eye(shape[1]), Z.T @ F, assume_a="pos")
    b = F.mean(axis=0)
    explicit = np.linalg.norm(Z @ W + b - F)**2 + gamma * np.linalg.norm(W)**2
    assert_allclose(np.trace(F.T @ Lg @ F), explicit, atol=1e-11)
    assert_allclose(Lg @ np.ones(len(X)), 0, atol=1e-12)
    assert linalg.eigvalsh(Lg).min() > -1e-12


@pytest.mark.parametrize("include_self", [False, True])
def test_local_elimination_equals_independently_solved_affine_regressions(include_self):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(27, 8)) + 10
    F = rng.normal(size=(27, 4))
    gamma, k = 0.8, 5
    L = local_regression_laplacian(X, k, gamma, include_self)
    _, neighbors = neighbor_indices(X, k - 1 if include_self else k)
    if include_self:
        neighbors = np.column_stack((np.arange(len(X)), neighbors))
    total = 0
    for ids in neighbors:
        # Solve original regression jointly in W and the UNPENALIZED intercept.
        design = np.column_stack((X[ids], np.ones(k)))
        penalty = np.diag([gamma] * X.shape[1] + [0.0])
        beta = linalg.solve(design.T @ design + penalty, design.T @ F[ids], assume_a="pos")
        total += np.linalg.norm(design @ beta - F[ids])**2 + gamma * np.linalg.norm(beta[:-1])**2
    assert_allclose(np.trace(F.T @ (L @ F)), total, rtol=1e-10, atol=1e-10)
    assert_allclose(L.toarray(), L.toarray().T, atol=1e-12)
    assert_allclose(L @ np.ones(len(X)), 0, atol=1e-12)
    assert linalg.eigvalsh(L.toarray()).min() > -1e-12


def test_self_inclusive_patches_cover_outlier():
    X = np.vstack((np.arange(10)[:, None], [[1000.0]]))
    excluded = local_regression_laplacian(X, k=3, include_self=False)
    included = local_regression_laplacian(X, k=3, include_self=True)
    assert excluded[-1, -1] == 0
    assert included[-1, -1] > 0


def test_laplacians_are_translation_invariant():
    X = np.random.default_rng(12).normal(size=(24, 5))
    for builder in (sc_laplacian, local_regression_laplacian):
        assert_allclose(builder(X).toarray(), builder(X + 15).toarray(), atol=1e-12)


def test_duplicate_points_exclude_self_and_remain_finite():
    X = np.zeros((12, 4))
    _, indices = neighbor_indices(X, 5)
    assert not np.any(indices == np.arange(len(X))[:, None])
    L = sc_laplacian(X).toarray()
    assert np.isfinite(L).all()
    assert_allclose(L, L.T)
    assert linalg.eigvalsh(L).min() > -1e-12


def test_rotation_recovers_rotated_indicator_with_zero_loss():
    rng = np.random.default_rng(6)
    y = np.repeat(np.arange(4), [12, 13, 14, 15])
    indicator = np.eye(4)[y]
    F = indicator / np.sqrt(np.bincount(y))
    Q = linalg.qr(rng.normal(size=(4, 4)))[0]
    result = spectral_rotation(F @ Q, n_init=10)
    assert clustering_accuracy(y, result.labels) == 1
    assert abs(result.objective) < 1e-10
    assert_allclose(result.rotation.T @ result.rotation, np.eye(4), atol=1e-12)
    assert np.all(np.diff(result.history) <= 1e-10)
    assert result.converged


@pytest.mark.parametrize("variant", ["sc", "lr"])
def test_solution_stationarity_and_original_objective(variant):
    X = np.random.default_rng(5).normal(size=(45, 9))
    model = SpectralEmbeddedClustering(3, variant=variant, mu=0.3, n_init=4).fit(X)
    assert model.eigen_residual_ < 1e-12
    assert model.orthogonality_error_ < 1e-11
    explicit = np.linalg.norm(model.transform(X) - model.embedding_)**2 + np.linalg.norm(model.coef_)**2
    assert_allclose(explicit, model.global_objective_, atol=1e-11)
    assert_allclose(model.eigenvalues_.sum(), model.spectral_objective_, atol=1e-11)
    Z = X - X.mean(axis=0)
    W = linalg.solve(Z.T @ Z + np.eye(9), Z.T @ model.embedding_)
    assert_allclose(model.coef_, W, atol=1e-12)


def test_mu_zero_reproduces_sc_subspace_including_trivial_vector():
    X = np.random.default_rng(1).normal(size=(35, 5))
    L = sc_laplacian(X).toarray()
    values, F = linalg.eigh(L, subset_by_index=(0, 2))
    model = SpectralEmbeddedClustering(3, mu=0, n_init=2).fit(X)
    assert abs(model.eigenvalues_[0]) < 1e-12
    assert_allclose(model.eigenvalues_, values, atol=1e-12)
    assert_allclose(model.embedding_ @ model.embedding_.T, F @ F.T, atol=1e-10)


@pytest.mark.parametrize("variant", ["sc", "lr"])
def test_unseen_clusters_and_saved_inference(tmp_path, variant):
    X, y = make_blobs(n_samples=180, centers=3, n_features=12, cluster_std=0.25, random_state=8)
    train, test = np.arange(120), np.arange(120, 180)
    model = SpectralEmbeddedClustering(3, variant=variant, mu=1, n_init=5).fit(X[train])
    assert clustering_accuracy(y[train], model.labels_) == 1
    assert clustering_accuracy(y[test], model.predict(X[test])) == 1
    # Predict in separate batches must equal predict on all points (no test centering).
    assert_allclose(model.transform(X[test]), np.concatenate([model.transform(X[i:i+1]) for i in test]))
    model.save(tmp_path / "model.npz")
    with np.load(tmp_path / "model.npz", allow_pickle=False) as state:
        Fnew = (X[test] - state["mean"]) @ state["coef"] + state["embedding_bias"]
        restored = (normalize_rows(Fnew) @ state["rotation"]).argmax(axis=1)
    assert_allclose(restored, model.predict(X[test]))
    loaded = SpectralEmbeddedClustering.load(tmp_path / "model.npz")
    assert_allclose(loaded.predict(X[test]), model.predict(X[test]))
    assert loaded.variant == variant and loaded.mu == model.mu


def test_acc_is_permutation_invariant_but_train_mapping_is_frozen():
    true, pred = np.array([5, 5, 7, 7]), np.array([1, 1, 0, 0])
    mapping = fit_label_mapping(true, pred)
    assert clustering_accuracy(true, pred) == 1
    assert mapped_accuracy(true, 1 - pred, mapping) == 0
    assert clustering_accuracy(true, 1 - pred) == 1


def test_deterministic_fit_and_invalid_inputs():
    X = np.random.default_rng(7).normal(size=(40, 6))
    first = SpectralEmbeddedClustering(3, mu=.1, n_init=3).fit(X)
    second = SpectralEmbeddedClustering(3, mu=.1, n_init=3).fit(X)
    assert_allclose(first.labels_, second.labels_)
    with pytest.raises(ValueError):
        SpectralEmbeddedClustering(3, mu=-1).fit(X)
    with pytest.raises(ValueError):
        SpectralEmbeddedClustering(3, gamma_global=0).fit(X)
    with pytest.raises(ValueError):
        first.predict(np.zeros((3, 5)))
    with pytest.raises(RuntimeError):
        SpectralEmbeddedClustering(3).predict(X)
