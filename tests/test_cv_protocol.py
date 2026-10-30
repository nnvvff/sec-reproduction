import json
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import run_cv_benchmark as runner
from sec.model import SpectralEmbeddedClustering, local_regression_laplacian
from sec.selection import TrainPreprocessor, select_candidate


def options(**overrides):
    values = dict(n_init=2, scale_neighbor=7, lr_include_self=True, kernel_center=True,
                  rbf_gamma="mean-distance", cv_folds=3, cv_values=[.01, 1.],
                  preprocessing_grid=["standard"], kernels=["linear", "cosine", "rbf"])
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_neighborhood_and_old_explicit_model_roundtrip(tmp_path):
    X = np.vstack((np.arange(10)[:, None], [[1000.]]))
    assert local_regression_laplacian(X, k=3)[-1, -1] > 0
    assert SpectralEmbeddedClustering(2).lr_include_self is True
    model = SpectralEmbeddedClustering(2, variant="lr", n_init=2, lr_include_self=False).fit(X)
    model.save(tmp_path / "old.npz")
    loaded = SpectralEmbeddedClustering.load(tmp_path / "old.npz")
    assert loaded.lr_include_self is False
    assert_allclose(loaded.predict(X), model.predict(X))


@pytest.mark.parametrize("mode", ["raw", "l2", "standard"])
def test_preprocessing_train_only_batch_invariance_and_reload(tmp_path, mode):
    X = np.array([[1., 2., 7.], [2., 3., 7.], [3., 4., 7.]])
    unseen = np.array([[100., 200., 700.], [-30., 40., 7.]])
    pre = TrainPreprocessor(mode).fit(X)
    before = pre.mean_.copy()
    whole = pre.transform(unseen)
    assert_allclose(whole, np.vstack([pre.transform(row[None]) for row in unseen]))
    assert_allclose(pre.mean_, before)
    if mode == "standard":
        assert_allclose(pre.mean_, X.mean(axis=0))
        assert_allclose(pre.scale_[-1], 1)
    pre.save(tmp_path / "pre.npz")
    assert_allclose(TrainPreprocessor.load(tmp_path / "pre.npz").transform(unseen), whole)


def test_cv_tie_rule_and_missing_fold_rejection():
    chosen, means = select_candidate([[.4, .6], [.5, .5], [.2, .4]])
    assert chosen == 0
    assert_allclose(means, [.5, .5, .3])
    for invalid in ([[.1, np.nan]], [[.2, float("inf")]], [[1.1, .2]], []):
        with pytest.raises(ValueError):
            select_candidate(invalid)


def test_candidates_preserve_predeclared_order_and_all_kernels():
    candidates = runner.candidates_for("CLGR", "optdigits", options(preprocessing_grid=["raw", "l2"]))
    assert len(candidates) == 12
    assert candidates[0] == dict(method="CLGR", parameter=.01, preprocessing="raw", kernel="linear")
    assert candidates[-1] == dict(method="CLGR", parameter=1., preprocessing="l2", kernel="rbf")


def test_seen_only_cv_artifacts_resume_and_tamper_rejection(tmp_path, monkeypatch):
    X = np.random.default_rng(12).normal(size=(36, 6))
    y = np.arange(36) % 3
    ids = np.arange(36) * 2 + 9
    args = options()
    candidates = runner.candidates_for("SEC/LR", "optdigits", args)
    protocol = dict(clusters=3, mu_lr=.01, mu_sc=.01, dkm_gamma=1., clgr_eta=1.)
    selected = runner.select_on_seen(X, y, ids, candidates, protocol, args, 42, tmp_path)
    assert selected["fit_count"] == 6
    all_validation = []
    for fold in tmp_path.glob("fold_*"):
        with np.load(fold / "split.npz") as state:
            assert not np.intersect1d(state["train"], state["validation"]).size
            assert_allclose(state["train_global"], ids[state["train"]])
            all_validation.extend(state["validation"])
    assert sorted(all_validation) == list(range(len(X)))
    # A clean resume cannot execute another model fit.
    monkeypatch.setattr(runner, "build_model", lambda *a, **k: pytest.fail("resume fitted a model"))
    assert runner.select_on_seen(X, y, ids, candidates, protocol, args, 42, tmp_path) == selected
    path = tmp_path / "fold_00/candidate_000/result.json"
    record = json.loads(path.read_text())
    record["test_acc"] = 1. - record["test_acc"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="metrics"):
        runner.select_on_seen(X, y, ids, candidates, protocol, args, 42, tmp_path)


def test_benchmark_kernel_variants_build_without_spurious_rbf_settings():
    from run_benchmark import build_model
    protocol = dict(clusters=3, clgr_eta=1.)
    X = np.random.default_rng(4).normal(size=(30, 5))
    for method in ("LL-linear", "LL-cosine", "LL-rbf", "CLGR-linear", "CLGR-cosine", "CLGR-rbf"):
        model = build_model(method, protocol, 4, options()).fit(X)
        assert model.kernel == method.split("-")[1]
