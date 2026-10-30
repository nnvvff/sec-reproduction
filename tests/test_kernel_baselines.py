"""Independent checks for explicit LL/CLGR kernel and fitted-scale settings."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from sec.baselines import (LocalLearningClustering, local_learning_operator,
                           resolve_local_rbf_gamma, self_tuning_affinity)


@pytest.mark.parametrize("kernel", ["linear", "cosine", "rbf"])
def test_local_operator_matches_independent_kernel_ridge(kernel):
    X = np.random.default_rng(109).normal(size=(31, 7)) + 2
    X[0] = 0  # Explicitly exercise the documented zero-row cosine convention.
    distance = cdist(X, X)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1)[:, :4]
    if kernel == "cosine":
        norms = np.linalg.norm(X, axis=1)
        features = np.divide(X, norms[:, None], out=np.zeros_like(X), where=norms[:, None] != 0)
        gram = features @ features.T
    elif kernel == "rbf":
        gram = np.exp(-.13 * cdist(X, X, "sqeuclidean"))
    else:
        gram = X @ X.T
    expected = np.zeros((len(X), len(X)))
    for i, ids in enumerate(neighbors):
        expected[i, ids] = solve(gram[np.ix_(ids, ids)] + .7*np.eye(4), gram[ids, i])
    N = local_learning_operator(X, k=4, gamma=.7, kernel=kernel,
                                rbf_gamma=.13 if kernel == "rbf" else None)
    assert_allclose(N.toarray(), expected, rtol=1e-12, atol=1e-13)
    assert_array_equal(N.diagonal(), 0)


def test_rbf_rules_match_definitions_and_reject_undefined_scale():
    X = 50 + np.random.default_rng(10).normal(size=(29, 7))
    assert_allclose(resolve_local_rbf_gamma(X, "mean-distance"),
                    1 / cdist(X, X, "sqeuclidean").mean(), rtol=1e-14)
    assert_allclose(resolve_local_rbf_gamma(X, "ll2006-mean-norm"),
                    1 / np.mean(np.linalg.norm(X, axis=1))**2, rtol=1e-14)
    # Mean-distance translation invariance; mean-norm intentionally differs.
    assert_allclose(resolve_local_rbf_gamma(X - 50, "mean-distance"),
                    resolve_local_rbf_gamma(X, "mean-distance"), rtol=1e-14)
    assert resolve_local_rbf_gamma(X - 50, "ll2006-mean-norm") > resolve_local_rbf_gamma(X, "ll2006-mean-norm")
    for value in (None, 0, -1, np.inf, True, "auto", "nan"):
        with pytest.raises(ValueError):
            resolve_local_rbf_gamma(X, value)
    for rule in ("mean-distance", "ll2006-mean-norm"):
        with pytest.raises(ValueError, match="undefined"):
            resolve_local_rbf_gamma(np.zeros((5, 3)), rule)


@pytest.mark.parametrize("variant", ["ll", "clgr"])
def test_fitted_kernel_scale_objective_prediction_and_metadata(variant, tmp_path):
    X = np.random.default_rng(46).normal(size=(38, 8)) + 5
    with threadpool_limits(limits=2):
        model = LocalLearningClustering(3, variant=variant, kernel="rbf",
            rbf_gamma="mean-distance", center=True, n_neighbors=4, eta=2.4, n_init=4).fit(X)
    expected_gamma = 1 / cdist(X, X, "sqeuclidean").mean()
    assert_allclose(model.rbf_gamma_, expected_gamma)
    Z, F = X - X.mean(axis=0), model.embedding_
    N = local_learning_operator(Z, k=4, gamma=1, kernel="rbf", rbf_gamma=expected_gamma)
    expected_objective = np.linalg.norm(N @ F - F)**2
    if variant == "clgr":
        A = self_tuning_affinity(Z, k=4).toarray()
        expected_objective = 2.4*expected_objective + .5*np.sum(A * cdist(F, F, "sqeuclidean"))
    assert_allclose(model.spectral_objective_, expected_objective, atol=1e-11)
    config = model.kernel_config_.copy()
    mean = model.kernel_mean_.copy()
    heldout = X[:5] + 100
    prediction = model.predict(heldout)
    assert_array_equal(prediction, np.concatenate([model.predict(row[None, :]) for row in heldout]))
    model.predict(heldout * 1e5)
    assert model.kernel_config_ == config
    assert_array_equal(model.kernel_mean_, mean)
    assert model.rbf_gamma_ == config["resolved_rbf_gamma"]
    assert config["fit_samples"] == len(X)
    path = tmp_path / f"{variant}.npz"
    model.save(path)
    restored = LocalLearningClustering.load(path)
    assert restored.kernel_config_ == config
    assert restored.rbf_gamma == "mean-distance"
    assert restored.variant == variant
    assert_array_equal(restored.predict(heldout), prediction)
    with np.load(path, allow_pickle=False) as state:
        assert set(("mean", "coef", "embedding_bias", "rotation", "labels", "embedding", "eigenvalues")) <= set(state.files)
        assert json.loads(str(state["configuration_json"]))["parameters"]["kernel"] == "rbf"


def test_default_linear_model_matches_explicit_legacy_settings(tmp_path):
    X = np.random.default_rng(72).normal(size=(30, 5))
    default = LocalLearningClustering(3, n_init=3).fit(X)
    explicit = LocalLearningClustering(3, kernel="linear", rbf_gamma=None,
                                       center=True, n_init=3).fit(X)
    assert_allclose(default.embedding_, explicit.embedding_, atol=0)
    assert_array_equal(default.labels_, explicit.labels_)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, mean=default.mean_, coef=default.coef_,
        embedding_bias=default.embedding_bias_, rotation=default.rotation_,
        labels=default.labels_, embedding=default.embedding_, eigenvalues=default.eigenvalues_)
    with pytest.warns(UserWarning, match="Legacy model"):
        restored = LocalLearningClustering.load(path)
    assert restored.kernel_config_ == {"source": "legacy_unrecorded"}
    assert_array_equal(default.predict(X + .1), restored.predict(X + .1))


def test_unused_rbf_settings_are_rejected_instead_of_silently_ignored():
    X = np.random.default_rng(7).normal(size=(20, 4))
    for kernel in ("linear", "cosine"):
        with pytest.raises(ValueError, match="only applies"):
            LocalLearningClustering(3, kernel=kernel, rbf_gamma=.1).fit(X)
