"""Compare fold reuse against independent ordinary model.fit calls."""

import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from sec.baselines import DiscriminativeKMeans, LocalLearningClustering, SpectralKMeans
from sec.cv_cache import FoldCache
from sec.model import SpectralEmbeddedClustering
from sec.selection import TrainPreprocessor


def models():
    common = dict(n_clusters=3, n_init=4, random_state=12)
    return [
        SpectralEmbeddedClustering(**common, variant="sc", mu=0),
        SpectralEmbeddedClustering(**common, variant="sc", mu=1e6),
        SpectralEmbeddedClustering(**common, variant="lr", mu=.07, lr_include_self=True),
        DiscriminativeKMeans(**common, gamma=.03),
        DiscriminativeKMeans(**common, gamma=10, prediction_metric="discriminative"),
        SpectralKMeans(**common),
        SpectralKMeans(**common, include_constant=True),
        KMeans(**common, init="random"),
    ] + [LocalLearningClustering(**common, variant=variant, kernel=kernel,
        rbf_gamma="mean-distance" if kernel == "rbf" else None, center=center, eta=3.7)
        for variant in ("ll", "clgr") for kernel in ("linear", "cosine", "rbf")
        for center in (True, False)]


@pytest.mark.parametrize("preprocessing", ["raw", "l2", "standard"])
def test_cached_models_match_original_training_prediction_and_objectives(preprocessing, tmp_path):
    rng = np.random.default_rng(709)
    X = rng.normal(size=(51, 9)) * np.arange(1, 10) + 50
    evaluation = rng.normal(size=(13, 9)) * np.arange(1, 10) + 52
    train = TrainPreprocessor(preprocessing).fit(X)
    Xt, Xe = train.transform(X), train.transform(evaluation)
    cache = FoldCache(X, preprocessing=preprocessing, fold_id="inner_0")
    assert_allclose(cache.X, Xt, atol=0, rtol=0)
    assert_allclose(cache.preprocessor.mean_, train.mean_, atol=0, rtol=0)
    assert_allclose(cache.preprocessor.scale_, train.scale_, atol=0, rtol=0)
    with threadpool_limits(limits=1):
        for i, template in enumerate(models()):
            direct = copy.deepcopy(template).fit(Xt)
            cached = cache.fit(copy.deepcopy(template))
            assert type(cached) is type(direct)
            assert_array_equal(cached.labels_, direct.labels_)
            assert_array_equal(cached.predict(Xe), direct.predict(Xe))
            for key in ("embedding_", "eigenvalues_", "coef_", "mean_", "rotation_", "intercept_",
                        "metric_projection_", "cluster_centers_", "metric_centers_",
                        "spectral_objective_", "graph_objective_", "global_objective_", "inertia_",
                        "eigen_residual_", "boundary_eigengap_", "residual_to_gap_", "trace_objective_"):
                if hasattr(direct, key):
                    assert_allclose(getattr(cached, key), getattr(direct, key), atol=1e-12, rtol=1e-12,
                                    err_msg=f"{type(direct).__name__}.{key}")
            if hasattr(direct, "kernel_config_"):
                assert cached.kernel_config_ == direct.kernel_config_
            if hasattr(cached, "save"):
                a, b = tmp_path/f"direct_{i}.npz", tmp_path/f"cached_{i}.npz"
                direct.save(a)
                cached.save(b)
                with np.load(a, allow_pickle=False) as expected, np.load(b, allow_pickle=False) as actual:
                    assert expected.files == actual.files
                    for key in expected.files:
                        if expected[key].dtype.kind in "biufc":
                            assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
                        else:
                            assert_array_equal(actual[key], expected[key])
    assert cache.stats["svd_once"] == 1
    assert cache.stats["svd_twice"] == 1
    assert cache.stats["global_laplacian"] == 1
    assert cache.stats["sc_laplacian"] == 1
    assert cache.stats["lr_laplacian"] == 1
    assert cache.stats["local_learning_operator"] == 6  # 3 kernels x 2 center settings
    assert cache.stats["clgr_laplacian"] == 2


def test_fold_cache_copies_training_data_and_cannot_cross_contaminate():
    rng = np.random.default_rng(32)
    first = rng.normal(size=(40, 8))
    second = rng.normal(size=(40, 8)) + 200
    owned = first.copy()
    one = FoldCache(first, "standard", fold_id="same label does not share state")
    two = FoldCache(second, "standard", fold_id="same label does not share state")
    first[:] = 1e9
    assert_allclose(one.preprocessor.mean_, owned.mean(axis=0))
    assert_allclose(two.preprocessor.mean_, second.mean(axis=0))
    with pytest.raises(ValueError):
        one.X[0, 0] = 0
    template = LocalLearningClustering(3, kernel="rbf", rbf_gamma="ll2006-mean-norm", n_init=3)
    with threadpool_limits(limits=1):
        m1, m2 = one.fit(copy.deepcopy(template)), two.fit(copy.deepcopy(template))
    assert one.stats["svd_once"] == two.stats["svd_once"] == 1
    assert m1.rbf_gamma_ != m2.rbf_gamma_
    before = one.stats.copy()
    m1.predict(one.preprocessor.transform(second * 1e5))
    assert one.stats == before
    assert one._local is not two._local
    assert one._svds is not two._svds


def test_config_changes_invalidate_only_relevant_cached_matrices():
    X = np.random.default_rng(62).normal(size=(44, 8))
    cache = FoldCache(X)
    with threadpool_limits(limits=1):
        for mu in (.01, 1, 100):
            cache.fit(SpectralEmbeddedClustering(3, mu=mu, n_init=2))
        assert cache.stats["sc_laplacian"] == cache.stats["global_laplacian"] == cache.stats["svd_twice"] == 1
        cache.fit(SpectralEmbeddedClustering(3, mu=1, gamma_global=2, scale_neighbor=3, n_init=2))
        assert cache.stats["sc_laplacian"] == cache.stats["global_laplacian"] == 2
        assert cache.stats["svd_twice"] == 1
        for eta in (.01, 1, 100):
            cache.fit(LocalLearningClustering(3, variant="clgr", eta=eta,
                kernel="rbf", rbf_gamma=.1, n_init=2))
        assert cache.stats["local_learning_operator"] == cache.stats["clgr_laplacian"] == cache.stats["svd_once"] == 1
        cache.fit(LocalLearningClustering(3, variant="clgr", kernel="rbf", rbf_gamma=.2, n_init=2))
        assert cache.stats["local_learning_operator"] == 2
        assert cache.stats["clgr_laplacian"] == 1
        cache.fit(LocalLearningClustering(3, kernel="linear", gamma_local=2, n_init=2))
        assert cache.stats["local_learning_operator"] == 3


def test_cache_rejects_invalid_parameters_and_keeps_zero_scale_rule():
    cache = FoldCache(np.random.default_rng(5).normal(size=(20, 5)))
    with pytest.raises(TypeError):
        cache.fit(object())
    for model in (SpectralEmbeddedClustering(20), SpectralEmbeddedClustering(3, mu=-1),
                  SpectralEmbeddedClustering(3, gamma_global=0), DiscriminativeKMeans(3, gamma=-1),
                  LocalLearningClustering(3, kernel="rbf"),
                  LocalLearningClustering(3, kernel="linear", rbf_gamma=.1)):
        with pytest.raises(ValueError):
            cache.fit(model)
    with pytest.raises(ValueError, match="undefined"):
        FoldCache(np.ones((20, 5))).fit(LocalLearningClustering(3, kernel="rbf", rbf_gamma="mean-distance"))


def test_sec_retains_double_centering_for_large_offsets():
    # Exact retained numerical convention, not merging the distinct SVD inputs.
    X = 1e10 + np.random.default_rng(41).normal(size=(40, 7))
    cache = FoldCache(X)
    model = SpectralEmbeddedClustering(3, mu=2, variant="lr", n_init=3)
    with threadpool_limits(limits=1):
        direct = copy.deepcopy(model).fit(X)
        cached = cache.fit(model)
    assert_allclose(cached.embedding_, direct.embedding_, rtol=0, atol=0)
    assert_allclose(cached.coef_, direct.coef_, rtol=0, atol=0)
    assert_array_equal(cached.predict(X[:8]), direct.predict(X[:8]))
