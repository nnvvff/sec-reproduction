"""Auditable, resumable multi-dataset reproduction. No label-based selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

from sec.model import SpectralEmbeddedClustering
from sec.metrics import clustering_accuracy, fit_label_mapping, mapped_accuracy
from sec.checkpoints import atomic_npz, ensure_split, validate_completed_fit, file_sha256
from paper_protocol import PROTOCOL, METHOD_COLUMNS, SWEEP_VALUES

ROOT = Path(__file__).resolve().parent
METHODS = METHOD_COLUMNS + [f"{method}-{kernel}" for method in ("LL", "CLGR") for kernel in ("cosine", "rbf")]


def bandwidth_argument(value):
    if value in ("mean-distance", "ll2006-mean-norm"):
        return value
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a positive coefficient, mean-distance, or ll2006-mean-norm") from error
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Bandwidth coefficient must be finite and positive")
    return number


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, content):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def build_model(method, protocol, seed, args, parameter=None):
    common = dict(n_clusters=protocol["clusters"], n_init=args.n_init, random_state=seed)
    if method == "KMeans":
        return KMeans(**common, init="random", max_iter=300, tol=1e-4)
    if method in ("SC+ridge", "SEC/SC", "SEC/LR"):
        mu = 0 if method == "SC+ridge" else protocol["mu_lr" if method == "SEC/LR" else "mu_sc"]
        if parameter is not None and method != "SC+ridge":
            mu = parameter
        return SpectralEmbeddedClustering(**common, variant="lr" if method == "SEC/LR" else "sc",
                                          mu=mu, n_neighbors=5, scale_neighbor=args.scale_neighbor,
                                          lr_include_self=args.lr_include_self)
    from sec.baselines import SpectralKMeans, DiscriminativeKMeans, LocalLearningClustering
    if method == "KM-r":
        return SpectralKMeans(**common)
    if method == "DKM":
        return DiscriminativeKMeans(**common, gamma=protocol["dkm_gamma"] if parameter is None else parameter,
                                    prediction_metric="original")
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return LocalLearningClustering(**common, variant="ll" if method.startswith("LL-") else "clgr",
                                   eta=protocol["clgr_eta"] if parameter is None else parameter,
                                   n_neighbors=5, scale_neighbor=args.scale_neighbor, kernel=method.split("-")[1],
                                   rbf_gamma=getattr(args, "rbf_gamma", "mean-distance") if method.endswith("-rbf") else None,
                                   center=getattr(args, "kernel_center", True))


def metrics(y_train, y_test, pred_train, pred_test):
    mapping = fit_label_mapping(y_train, pred_train)
    return {"train_acc": clustering_accuracy(y_train, pred_train),
            "test_acc": clustering_accuracy(y_test, pred_test),
            "test_acc_frozen_mapping": mapped_accuracy(y_test, pred_test, mapping),
            "train_nmi": normalized_mutual_info_score(y_train, pred_train),
            "test_nmi": normalized_mutual_info_score(y_test, pred_test),
            "train_ari": adjusted_rand_score(y_train, pred_train),
            "test_ari": adjusted_rand_score(y_test, pred_test)}


def run_dataset(name, args):
    source = args.data_dir / f"{name}.npz"
    if not source.exists():
        raise FileNotFoundError(f"Missing exact dataset: {source}; run prepare_datasets.py / see DATASETS.md")
    protocol = PROTOCOL[name]
    with np.load(source, allow_pickle=False) as archive:
        X, y = np.asarray(archive["X"], dtype=float), archive["y"]
    if X.shape != protocol["shape"] or y.shape != (len(X),) or len(np.unique(y)) != protocol["clusters"]:
        raise ValueError(f"{name} does not match Table I: {X.shape}, {len(np.unique(y))} classes")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite feature values")
    if args.preprocessing == "l2":
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), np.finfo(float).eps)
    metadata_path = source.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    output = args.output / name
    output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "run_benchmark.py", ROOT / "paper_protocol.py", ROOT / "requirements.txt",
               *sorted((ROOT / "sec").glob("*.py"))]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    config = {"dataset": name, "data_sha256": sha256(source), "data_metadata": metadata,
              "protocol": protocol, "splits": args.splits, "n_init": args.n_init,
              "seed": args.seed, "methods": args.methods, "preprocessing": args.preprocessing,
              "lr_include_self": args.lr_include_self, "scale_neighbor": args.scale_neighbor,
              "rbf_gamma": getattr(args, "rbf_gamma", "mean-distance"),
              "kernel_center": getattr(args, "kernel_center", True),
              "sweep": args.sweep, "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "sklearn": sklearn.__version__, "source_sha256": hashes,
              "threads": args.threads, "parameter_selection": "all sweep values reported; no label-based selection"
              if args.sweep else "fixed Table IV values; no CV retuning"}
    # JSON roundtrip makes tuples comparable across resume.
    config = json.loads(json.dumps(config))
    config_path = output / "config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"{output} exists: explicitly --resume or use a new output")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume rejected: configuration, data, environment or source changed")
        if any(file_sha256(output / "source" / relative) != digest for relative, digest in hashes.items()):
            raise ValueError("Resume rejected: source snapshot checksum mismatch")
    else:
        for path in sources:
            destination = output / "source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        atomic_json(config_path, config)
    failures = []
    start_total = time.perf_counter()
    expected = sum((len(SWEEP_VALUES[:7] if name == "mnist" else SWEEP_VALUES)
                    if args.sweep and (method in ("SEC/SC", "SEC/LR", "DKM") or method.startswith("CLGR-")) else 1)
                   for method in args.methods) * args.splits
    fatal_error = None
    atomic_json(output / "status.json", {"state": "running", "expected_fits": expected})
    try:
        with threadpool_limits(limits=args.threads):
            for split in range(args.splits):
                seed = args.seed + split
                ordering = np.random.default_rng(seed).permutation(len(X))
                n_seen = int(len(X) * protocol["fraction"])
                train, test = ordering[:n_seen], ordering[n_seen:]
                split_dir = output / f"split_{split:02d}"
                split_dir.mkdir(exist_ok=True)
                ensure_split(split_dir / "split.npz", train=train, test=test,
                             y_train=y[train], y_test=y[test])
                for method in args.methods:
                    grid = SWEEP_VALUES[:7] if name == "mnist" else SWEEP_VALUES
                    values = grid if args.sweep and (method in ("SEC/SC", "SEC/LR", "DKM") or method.startswith("CLGR-")) else [None]
                    for parameter in values:
                        slug = method.replace("/", "_") + (f"_p{parameter:.0e}" if parameter is not None else "")
                        fit_dir = split_dir / slug
                        fit_dir.mkdir(exist_ok=True)
                        identity = dict(dataset=name, method=method, split=split, seed=seed, parameter=parameter)
                        if validate_completed_fit(fit_dir, identity, len(train), len(test)):
                            (fit_dir / "failure.json").unlink(missing_ok=True)
                            continue
                        started = time.perf_counter()
                        try:
                            model = build_model(method, protocol, seed, args, parameter).fit(X[train])
                            pred_train = np.asarray(model.labels_)
                            pred_test = model.predict(X[test])
                            # Evaluation comes strictly after all fitting/discretization decisions.
                            record = dict(dataset=name, method=method, split=split, seed=seed,
                                          parameter=parameter, **metrics(y[train], y[test], pred_train, pred_test),
                                          seconds=time.perf_counter()-started,
                                          clusters_found=int(len(np.unique(pred_train))))
                            for attr in ("eigen_residual_", "orthogonality_error_", "rotation_converged_",
                                         "rotation_objective_", "spectral_objective_", "boundary_eigengap_",
                                         "residual_to_gap_", "zero_local_diagonal_count_", "inertia_"):
                                if hasattr(model, attr):
                                    value = getattr(model, attr)
                                    record[attr.removesuffix("_")] = value.item() if isinstance(value, np.generic) else value
                            if hasattr(model, "eigenvalues_"):
                                record["eigenvalues"] = model.eigenvalues_.tolist()
                            if hasattr(model, "kernel_config_"):
                                record["kernel_config"] = model.kernel_config_
                            if hasattr(model, "eigenpair_absolute_residuals_"):
                                record["eigenpair_absolute_residuals"] = model.eigenpair_absolute_residuals_.tolist()
                            if method == "KMeans":
                                atomic_npz(fit_dir / "model.npz", centers=model.cluster_centers_)
                            else:
                                model.save(fit_dir / "model.tmp.npz")
                                (fit_dir / "model.tmp.npz").replace(fit_dir / "model.npz")
                            atomic_npz(fit_dir / "predictions.npz", train=pred_train, test=pred_test)
                            record["artifact_sha256"] = {filename: file_sha256(fit_dir / filename)
                                                         for filename in ("model.npz", "predictions.npz")}
                            atomic_json(fit_dir / "result.json", record)
                            (fit_dir / "failure.json").unlink(missing_ok=True)
                            print(f"{name} {split+1:02}/{args.splits} {slug} "
                                  f"train={record['train_acc']:.4f} test={record['test_acc']:.4f} "
                                  f"{record['seconds']:.1f}s", flush=True)
                        except Exception as error:
                            failure = dict(dataset=name, method=method, split=split, parameter=parameter,
                                           error=repr(error), traceback=traceback.format_exc())
                            atomic_json(fit_dir / "failure.json", failure)
                            failures.append(failure)
                            print(f"FAILED {name}/{split}/{slug}: {error}", flush=True)
                            if not args.continue_on_error:
                                raise
    except BaseException as error:
        fatal_error = repr(error)
        raise
    finally:
        records = []
        for path in sorted(output.glob("split_*/*/result.json")):
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError) as error:
                failures.append({"path": str(path), "error": repr(error)})
        atomic_json(output / "runs.json", records)
        state = "failed" if fatal_error or failures else "complete" if len(records) == expected else "incomplete"
        atomic_json(output / "status.json", {"state": state, "n_fits": len(records), "expected_fits": expected,
                                             "failures": failures, "fatal_error": fatal_error,
                                             "seconds_this_invocation": time.perf_counter()-start_total})
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(PROTOCOL), default=["optdigits"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["KMeans", "SC+ridge", "SEC/SC", "SEC/LR"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmark_v2")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scale-neighbor", type=int, default=7)
    parser.add_argument("--preprocessing", choices=["raw", "l2"], default="raw")
    parser.add_argument("--lr-include-self", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rbf-gamma", type=bandwidth_argument, default="mean-distance",
                        help="RBF coefficient or a training-only scale convention; applies only to RBF kernels")
    parser.add_argument("--kernel-center", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sweep", action="store_true", help="Report every Figure 1 grid value without selecting on test labels")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if min(args.splits, args.n_init, args.threads, args.scale_neighbor) < 1:
        parser.error("counts must be positive")
    if len(set(args.methods)) != len(args.methods) or len(set(args.datasets)) != len(args.datasets):
        parser.error("duplicate methods/datasets are not allowed")
    failures = 0
    for dataset in args.datasets:
        failures += run_dataset(dataset, args)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
