import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from threadpoolctl import threadpool_limits

from sec.baselines import (DiscriminativeKMeans, LocalLearningClustering, SpectralKMeans,
                           local_learning_operator, self_tuning_affinity)
from sec.metrics import clustering_accuracy
from sec.model import normalize_rows
from sec.nystrom import NystromSpectralClustering, nystrom_embedding_from_blocks, _squared_distances


def test_dkm_factorization_and_discrete_objective_match_eq34():
    X = np.random.default_rng(4).normal(size=(50, 8)) + 9
    with threadpool_limits(limits=2):
        model = DiscriminativeKMeans(3, gamma=.7, n_init=5).fit(X)
    Z = X-X.mean(axis=0)
    K = Z @ linalg.solve(Z.T @ Z + .7*np.eye(8), Z.T, assume_a="pos")
    P = model.transform(X)
    assert_allclose(P @ P.T, K, atol=1e-12)
    G = np.eye(3)[model.labels_] / np.sqrt(np.bincount(model.labels_))
    assert_allclose(np.trace(G.T @ K @ G), model.trace_objective_, atol=1e-11)
    assert_allclose(np.trace(K)-np.trace(G.T @ K @ G), model.inertia_, atol=1e-11)
    new = X[:7] + .1
    assert_allclose(model.predict(new), cdist(new, model.cluster_centers_).argmin(axis=1))


def test_kmr_matches_eq27_and_constant_variant_is_explicit():
    X = np.random.default_rng(3).normal(size=(40, 9))
    Z = X - X.mean(axis=0)
    vals, U = linalg.eigh(Z @ Z.T)
    model = SpectralKMeans(3, n_init=5).fit(X)
    expected = U[:, -3:]
    assert_allclose(model.embedding_ @ model.embedding_.T, expected @ expected.T, atol=1e-12)
    assert_allclose(model.embedding_.T @ np.ones(len(X)), 0, atol=1e-12)
    augmented = SpectralKMeans(3, n_init=5, include_constant=True).fit(X)
    assert_allclose(augmented.embedding_[:, 0], np.ones(len(X))/np.sqrt(len(X)))
    assert augmented.eigen_residual_ < 1e-12


def test_ll_operator_matches_direct_primal_ridge_predictions():
    rng = np.random.default_rng(11)
    X, F = rng.normal(size=(25, 7)), rng.normal(size=(25, 3))
    N = local_learning_operator(X, k=4, gamma=1.2)
    distance = cdist(X, X)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1)[:, :4]
    expected = []
    for i, ids in enumerate(neighbors):
        W = linalg.solve(X[ids].T @ X[ids] + 1.2*np.eye(7), X[ids].T @ F[ids], assume_a="pos")
        expected.append(X[i] @ W)
    assert_allclose(N @ F, expected, atol=1e-12)
    assert_allclose(N.diagonal(), 0)
    explicit = np.linalg.norm(np.array(expected)-F)**2
    R = np.eye(len(X))-N.toarray()
    assert_allclose(np.trace(F.T @ R.T @ R @ F), explicit, atol=1e-11)


@pytest.mark.parametrize("variant", ["ll", "clgr"])
def test_ll_clgr_original_objective_and_predict(variant):
    X = np.random.default_rng(16).normal(size=(35, 6))
    model = LocalLearningClustering(3, variant=variant, eta=2.1, n_init=4).fit(X)
    F, Z = model.embedding_, X-X.mean(axis=0)
    N = local_learning_operator(Z)
    local_loss = np.linalg.norm(N @ F-F)**2
    explicit = local_loss
    if variant == "clgr":
        A = self_tuning_affinity(Z).toarray()
        pairwise = np.sum((F[:,None,:]-F[None,:,:])**2, axis=2)
        explicit = .5*np.sum(A*pairwise) + 2.1*local_loss
    assert_allclose(explicit, model.spectral_objective_, atol=1e-11)
    assert model.eigen_residual_ < 1e-12
    assert model.orthogonality_error_ < 1e-12
    assert_allclose(model.transform(X[:4]), np.vstack([model.transform(x[None,:]) for x in X[:4]]))


def test_kernel_local_learning_rbf_needs_explicit_bandwidth():
    X = np.random.default_rng(1).normal(size=(15, 3))
    with pytest.raises(ValueError):
        local_learning_operator(X, kernel="rbf")
    N = local_learning_operator(X, kernel="rbf", rbf_gamma=.2)
    assert np.isfinite(N.data).all()
    assert_allclose(N.diagonal(), 0)


def test_dkm_rejects_degenerate_empty_clusters():
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="fewer nonempty clusters"):
        DiscriminativeKMeans(3, n_init=1).fit(np.zeros((8, 2)))


def test_nystrom_orthogonalization_matches_full_block_eigendecomposition():
    rng = np.random.default_rng(8)
    # Positive low-rank features yield positive affinities and degrees.
    features = rng.uniform(.2, 1.0, size=(15, 4))
    W = features @ features.T
    A, B = W[:7, :7], W[:7, 7:]
    F, values, diagnostic = nystrom_embedding_from_blocks(A, B, 3)
    reconstructed = np.block([[A, B], [B.T, B.T @ linalg.pinvh(A) @ B]])
    degrees = reconstructed.sum(axis=1)
    K = reconstructed / np.sqrt(degrees[:,None]*degrees[None,:])
    expected_values, expected = linalg.eigh(K)
    assert_allclose(values, expected_values[-3:][::-1], atol=1e-12)
    assert_allclose(F @ F.T, expected[:,-3:] @ expected[:,-3:].T, atol=1e-11)
    assert_allclose(F.T @ F, np.eye(3), atol=1e-12)
    assert diagnostic["retained_rank"] == 4
    assert diagnostic["eigen_residual"] < 1e-12


def test_nystrom_gemm_distances_match_direct_differences_with_large_offset():
    rng = np.random.default_rng(21)
    X = 1e8 + rng.normal(size=(13, 2048))
    Y = 1e8 + rng.normal(size=(9, 2048))
    assert_allclose(_squared_distances(X, Y), cdist(X, Y, metric="sqeuclidean"), rtol=2e-14)
    assert_allclose(_squared_distances(X), cdist(X, X, metric="sqeuclidean"), rtol=2e-14)
    assert_allclose(_squared_distances(X).diagonal(), 0, atol=0)


def test_nystrom_truncation_projects_cross_block_as_documented():
    rng = np.random.default_rng(71)
    samples, rest = rng.uniform(.3, 1, (5, 2)), rng.uniform(.3, 1, (7, 2))
    A = samples @ samples.T + 1e-9*np.eye(5)
    B = samples @ rest.T + .01*rng.uniform(size=(5, 7))
    F, values, diagnostic = nystrom_embedding_from_blocks(A, B, 2, rcond=1e-5)
    vals, U = linalg.eigh(A)
    keep = vals > vals.max()*1e-5
    Q, lam = U[:, keep], vals[keep]
    Ar, Br = (Q*lam)@Q.T, Q@Q.T@B
    C = B.T @ (Q/lam) @ Q.T @ B
    W = np.block([[Ar, Br], [Br.T, C]])
    degree = W.sum(axis=1)
    K = W/np.sqrt(degree[:,None]*degree[None,:])
    expected_vals, expected = linalg.eigh(K)
    assert diagnostic["retained_rank"] == 2
    assert not np.allclose(Br, B, atol=1e-5)
    assert_allclose(values, expected_vals[-2:][::-1], atol=1e-12)
    assert_allclose(F@F.T, expected[:,-2:]@expected[:,-2:].T, atol=1e-11)


def test_baseline_saved_predictions_and_synthetic_unseen(tmp_path):
    X, y = make_blobs(n_samples=100, n_features=8, centers=3, cluster_std=.3, random_state=2)
    with threadpool_limits(limits=2):
        model = DiscriminativeKMeans(3, gamma=1, n_init=10).fit(X[:70])
        assert clustering_accuracy(y[70:],model.predict(X[70:])) == 1
        model.save(tmp_path / "dkm.npz")
        with np.load(tmp_path / "dkm.npz", allow_pickle=False) as state:
            assert_allclose(model.predict(X[70:]), cdist(X[70:],state["original_centers"]).argmin(axis=1))
        model = LocalLearningClustering(3, n_init=5).fit(X[:70])
        model.save(tmp_path / "ll.npz")
        with np.load(tmp_path / "ll.npz", allow_pickle=False) as state:
            F=(X[70:]-state["mean"])@state["coef"]+state["embedding_bias"]
            assert_allclose(model.predict(X[70:]),(normalize_rows(F)@state["rotation"]).argmax(axis=1))
        nystrom = NystromSpectralClustering(3, n_init=5).fit(X[:70], X[70:])
        assert clustering_accuracy(y[70:], nystrom.unseen_labels_) == 1
        assert nystrom.eigen_residual_ < 1e-12
