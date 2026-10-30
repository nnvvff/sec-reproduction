"""Nested seen-only CV with explicit, auditable replacement conventions.

Inner validation labels select parameters. Outer test features/labels are not
passed to selection. This is NOT the author's unrecovered CV implementation.
"""
import argparse
import itertools
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from paper_protocol import PROTOCOL, SWEEP_VALUES
from run_benchmark import atomic_json, build_model, bandwidth_argument, metrics
from sec.checkpoints import atomic_npz, ensure_split, file_sha256, validate_completed_fit
from sec.selection import TrainPreprocessor, select_candidate

ROOT = Path(__file__).resolve().parent
METHODS = ("SEC/SC", "SEC/LR", "DKM", "LL", "CLGR", "SC+ridge", "KM-r", "KMeans")
TUNED = ("SEC/SC", "SEC/LR", "DKM", "CLGR")


def candidates_for(method, dataset, args):
    values = args.cv_values or (SWEEP_VALUES[:7] if dataset == "mnist" else SWEEP_VALUES)
    values = values if method in TUNED else [None]
    kernels = args.kernels if method in ("LL", "CLGR") else [None]
    return [dict(method=method, parameter=value, preprocessing=preprocessing, kernel=kernel)
            for preprocessing, kernel, value in itertools.product(args.preprocessing_grid, kernels, values)]


def fit_record(X_train, y_train, X_eval, y_eval, candidate, protocol, args, seed, directory, identity, cache=None):
    """One committed fit; saved preprocessing is part of the inference artifact."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    expected = dict(identity, candidate=candidate, seed=int(seed))
    if validate_completed_fit(directory, expected, len(X_train), len(X_eval)):
        record = json.loads((directory / "result.json").read_text())
        if file_sha256(directory / "preprocessing.npz") != record["preprocessing_sha256"]:
            raise ValueError("Preprocessing checkpoint checksum mismatch")
        # Selection never trusts a stored scalar score without checking labels.
        with np.load(directory / "predictions.npz", allow_pickle=False) as pred:
            recomputed = metrics(y_train, y_eval, pred["train"], pred["test"])
        if any(not np.isclose(record[key], value, rtol=0, atol=1e-14) for key, value in recomputed.items()):
            raise ValueError("Saved metrics differ from saved predictions")
        (directory / "failure.json").unlink(missing_ok=True)
        return record
    start = time.perf_counter()
    try:
        preprocess = (cache.preprocessor if cache is not None
                      else TrainPreprocessor(candidate["preprocessing"]).fit(X_train))
        method = candidate["method"]
        if method in ("LL", "CLGR"):
            method += "-" + candidate["kernel"]
        model = build_model(method, protocol, seed, args, candidate["parameter"])
        if cache is None:
            model.fit(preprocess.transform(X_train))
        else:
            model = cache.fit(model)
        pred_train = np.asarray(model.labels_)
        pred_eval = model.predict(preprocess.transform(X_eval))
        record = dict(expected, **metrics(y_train, y_eval, pred_train, pred_eval),
                      seconds=time.perf_counter() - start, clusters_found=int(len(np.unique(pred_train))))
        if hasattr(model, "kernel_config_"):
            record["kernel_config"] = model.kernel_config_
        for attr in ("eigen_residual_", "residual_to_gap_", "rotation_converged_", "rotation_objective_", "inertia_"):
            if hasattr(model, attr):
                value = getattr(model, attr)
                record[attr.rstrip("_")] = value.item() if isinstance(value, np.generic) else value
        if candidate["method"] == "KMeans":
            atomic_npz(directory / "model.npz", centers=model.cluster_centers_)
        else:
            model.save(directory / "model.tmp.npz")
            (directory / "model.tmp.npz").replace(directory / "model.npz")
        preprocess.save(directory / "preprocessing.npz")
        atomic_npz(directory / "predictions.npz", train=pred_train, test=pred_eval)
        record["artifact_sha256"] = {name: file_sha256(directory / name) for name in ("model.npz", "predictions.npz")}
        record["preprocessing_sha256"] = file_sha256(directory / "preprocessing.npz")
        atomic_json(directory / "result.json", record)
        (directory / "failure.json").unlink(missing_ok=True)
        return record
    except BaseException as error:
        atomic_json(directory / "failure.json", dict(identity, error=repr(error)))
        raise


def select_on_seen(X_seen, y_seen, seen_indices, candidates, protocol, args, seed, directory):
    """Selection boundary deliberately accepts NO outer-test data or labels."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if args.cv_folds < 2 or args.cv_folds > len(X_seen):
        raise ValueError("CV requires 2 <= folds <= number of seen samples")
    folds = list(KFold(n_splits=args.cv_folds, shuffle=True, random_state=seed).split(X_seen))
    scores = np.empty((len(candidates), len(folds)))
    for fold, (train, validation) in enumerate(folds):
        if len(train) <= max(5, protocol["clusters"]):
            raise ValueError("Too few inner training samples for graph/cluster count")
        fold_dir = directory / f"fold_{fold:02d}"
        fold_dir.mkdir(exist_ok=True)
        ensure_split(fold_dir / "split.npz", train=train, validation=validation,
                     train_global=seen_indices[train], validation_global=seen_indices[validation],
                     y_train=y_seen[train], y_validation=y_seen[validation])
        # Same restart seed across candidates within a fold, independent of labels.
        fold_seed = int((seed + 104729 * (fold + 1)) % (2 ** 32))
        caches = {}
        for index, candidate in enumerate(candidates):
            cache = None
            if getattr(args, "cache", False):
                from sec.cv_cache import FoldCache
                mode = candidate["preprocessing"]
                if mode not in caches:
                    caches[mode] = FoldCache(X_seen[train], preprocessing=mode)
                cache = caches[mode]
            result = fit_record(X_seen[train], y_seen[train], X_seen[validation], y_seen[validation],
                                candidate, protocol, args, fold_seed, fold_dir / f"candidate_{index:03d}",
                                dict(stage="inner_validation", fold=fold, candidate_index=index), cache=cache)
            scores[index, fold] = result["test_acc"]
            print(f"CV fold={fold + 1}/{len(folds)} candidate={index + 1}/{len(candidates)} "
                  f"validation_ACC={scores[index, fold]:.4f}", flush=True)
    chosen, means = select_candidate(scores)
    selection = dict(protocol="seen-only-unstratified-kfold-validation-acc-v1", candidates=candidates,
                     validation_acc=scores.tolist(), mean_validation_acc=means.tolist(), chosen_index=chosen,
                     chosen_candidate=candidates[chosen], tie_break="first candidate in declared order",
                     fit_count=len(candidates) * len(folds), seed=seed,
                     note="Uses inner-validation labels; no outer-test data enters selection. Author CV details unrecovered.")
    selection_path = directory / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Saved selection differs from recomputed fold scores")
    atomic_json(selection_path, selection)
    return selection


def run_dataset(name, args):
    protocol = PROTOCOL[name]
    data_path = args.data_dir / f"{name}.npz"
    with np.load(data_path, allow_pickle=False) as data:
        X, y = np.asarray(data["X"], dtype=float), data["y"]
    if X.shape != protocol["shape"] or y.shape != (len(X),) or len(np.unique(y)) != protocol["clusters"]:
        raise ValueError("Dataset does not match Table I")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite features")
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    split_indices = getattr(args, "split_indices", None)
    split_indices = list(range(args.splits)) if split_indices is None else list(split_indices)
    if not split_indices or len(set(split_indices)) != len(split_indices) or any(s < 0 or s >= args.splits for s in split_indices):
        raise ValueError("split-indices must be unique entries in range(splits)")
    sources = [ROOT / filename for filename in ("run_cv_benchmark.py", "run_benchmark.py", "paper_protocol.py", "requirements.txt")]
    sources += sorted((ROOT / "sec").glob("*.py"))
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in sources}
    config = dict(dataset=name, data_sha256=file_sha256(data_path), protocol=protocol,
                  selection_protocol="seen-only-unstratified-kfold-validation-acc-v1", cv_folds=args.cv_folds,
                  candidates={method: candidates_for(method, name, args) for method in args.methods},
                  methods=args.methods, splits=args.splits, n_init=args.n_init, seed=args.seed, threads=args.threads,
                  split_indices=split_indices, cache=getattr(args, "cache", False),
                  lr_include_self=args.lr_include_self, scale_neighbor=args.scale_neighbor,
                  rbf_gamma=args.rbf_gamma, kernel_center=args.kernel_center, source_sha256=source_hashes,
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    config = json.loads(json.dumps(config))
    config_path = output / "config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume with identical config/source or a fresh directory")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume rejected: configuration/data/source/environment changed")
        if any(file_sha256(output / "source" / path) != digest for path, digest in source_hashes.items()):
            raise ValueError("Resume rejected: source snapshot checksum mismatch")
    else:
        for source in sources:
            destination = output / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        atomic_json(config_path, config)
    expected_inner = len(split_indices) * args.cv_folds * sum(map(len, config["candidates"].values()))
    expected_final = len(split_indices) * len(args.methods)
    state, error = "running", None
    atomic_json(output / "status.json", dict(state=state, expected_inner_fits=expected_inner, expected_final_fits=expected_final))
    try:
        with threadpool_limits(limits=args.threads):
            for split in split_indices:
                seed = args.seed + split
                order = np.random.default_rng(seed).permutation(len(X))
                n_seen = int(len(X) * protocol["fraction"])
                train, test = order[:n_seen], order[n_seen:]
                split_dir = output / f"split_{split:02d}"
                split_dir.mkdir(exist_ok=True)
                ensure_split(split_dir / "split.npz", train=train, test=test, y_train=y[train], y_test=y[test])
                for method in args.methods:
                    method_dir = split_dir / method.replace("/", "_")
                    selection = select_on_seen(X[train], y[train], train, config["candidates"][method],
                                               protocol, args, seed, method_dir / "cv")
                    # The only final fit/evaluation follows the completed selection.
                    record = fit_record(X[train], y[train], X[test], y[test], selection["chosen_candidate"],
                                        protocol, args, seed, method_dir / "final",
                                        dict(stage="outer_test", dataset=name, method=method, split=split))
                    print(f"FINAL {name}/{split}/{method} selected={selection['chosen_candidate']} "
                          f"test_ACC={record['test_acc']:.4f}", flush=True)
        state = "complete"
    except BaseException as caught:
        state, error = "failed", repr(caught)
        raise
    finally:
        rows, read_errors = [], []
        for path in sorted(output.glob("split_*/*/final/result.json")):
            try:
                rows.append(json.loads(path.read_text()))
            except (OSError, ValueError) as caught:
                read_errors.append(dict(path=str(path), error=repr(caught)))
        if read_errors:
            state = "failed"
        n_inner = sum(1 for _ in output.glob("split_*/*/cv/fold_*/candidate_*/result.json"))
        atomic_json(output / "runs.json", rows)
        atomic_json(output / "status.json", dict(state=state, error=error, read_errors=read_errors, inner_fits=n_inner, final_fits=len(rows),
                    expected_inner_fits=expected_inner, expected_final_fits=expected_final))
        write_report(output, rows, config, state)


def write_report(output, rows, config, state):
    lines = ["# 训练集内部交叉验证（替代协议）", "", f"数据集：{config['dataset']}；状态：{state}。",
             f"计划外层 {config['splits']} 次非分层划分；本分片负责 {config.get('split_indices', list(range(config['splits'])))}；内层 {config['cv_folds']} 折；每次拟合 {config['n_init']} 次初始化。",
             "内层验证 ACC 选参，确定后在全部 seen 上重新拟合一次并评估 unseen。所有预处理统计量只从对应训练折计算。",
             "这是明确的项目替代协议，不能声称恢复了论文未公开的交叉验证细节；单次划分不是论文20次结果。", "",
             "| 外层划分 | 方法 | 所选配置 | 训练 ACC (%) | 测试 ACC (%) |", "|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['method']} | `{json.dumps(row['candidate'], ensure_ascii=False)}` | "
                     f"{100 * row['train_acc']:.2f} | {100 * row['test_acc']:.2f} |")
    lines += ["", "cv/selection.json 保存完整候选/逐折分数/选择依据；fold_*/split.npz 保存局部与原始行索引。",
              "每个已提交拟合均保存模型、预处理状态、预测和SHA256；旧固定参数结果没有被覆盖。",
              "内层聚类ACC各折独立Hungarian匹配，不等于固定语义标签的分类验证准确率。",
              "训练/测试划分共享样本的多次实验与选参存在自身统计限制，不据此宣称原文相对性能或显著性已复现。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(PROTOCOL), default=["optdigits"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["SEC/SC", "SEC/LR"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--split-indices", nargs="+", type=int, help="Disjoint outer split shard; preserve global seed+split identity")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse training-only matrices within each inner fold; exact finite-parameter solves")
    parser.add_argument("--cv-values", nargs="+", type=float, help="Default: Figure 1 grid, fixed before any evaluation")
    parser.add_argument("--preprocessing-grid", nargs="+", choices=["raw", "l2", "standard"], default=["raw"])
    parser.add_argument("--kernels", nargs="+", choices=["linear", "cosine", "rbf"], default=["linear", "cosine", "rbf"])
    parser.add_argument("--rbf-gamma", type=bandwidth_argument, default="mean-distance")
    parser.add_argument("--kernel-center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr-include-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-neighbor", type=int, default=7)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.splits, args.n_init, args.threads, args.scale_neighbor) < 1 or args.cv_folds < 2:
        parser.error("Counts must be positive and cv-folds >= 2")
    for values in (args.datasets, args.methods, args.kernels, args.preprocessing_grid, args.cv_values or []):
        if len(values) != len(set(values)):
            parser.error("Duplicate values are not allowed")
    if args.cv_values is not None and any(not np.isfinite(value) or value <= 0 for value in args.cv_values):
        parser.error("CV values must be finite and positive")
    if not 0 <= args.seed < 2 ** 32 - args.splits:
        parser.error("Seed/split range must fit unsigned 32-bit CV random state")
    for dataset in args.datasets:
        run_dataset(dataset, args)


if __name__ == "__main__":
    main()
