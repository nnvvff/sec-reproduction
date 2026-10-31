"""Run MNIST clustering and write results from this run, with optional resume."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from sec import SpectralEmbeddedClustering
from sec.cv_cache import FoldCache
from sec.data import SHA256, load_mnist
from sec.metrics import clustering_accuracy

ROOT = Path(__file__).resolve().parent
METHODS = ("SEC/SC", "SEC/LR")
MU_VALUES = (1e-9, 1e-6, 1e-3, 1., 1e3, 1e6, 1e9)


def build_model(method, seed, n_init, mu=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return SpectralEmbeddedClustering(
        10, variant="lr" if method == "SEC/LR" else "sc",
        mu=mu,
        n_neighbors=5, scale_neighbor=7, lr_include_self=True,
        n_init=n_init, random_state=seed)


def select_mu(X_seen, y_seen, method, seed, n_init):
    """Select using inner-validation labels; accepts no outer-test data."""
    scores = np.empty((len(MU_VALUES), 3))
    for fold, (train, validation) in enumerate(
            KFold(3, shuffle=True, random_state=seed).split(X_seen)):
        fold_seed = int((seed + 104729 * (fold + 1)) % (2 ** 32))
        cache = FoldCache(X_seen[train])
        for candidate, mu in enumerate(MU_VALUES):
            model = cache.fit(build_model(method, fold_seed, n_init, mu))
            scores[candidate, fold] = clustering_accuracy(
                y_seen[validation], model.predict(X_seen[validation]))
        print(f"  {method}: inner fold {fold + 1}/3 complete", flush=True)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite CV score; no candidates may be silently dropped")
    means = scores.mean(axis=1)
    # np.argmax preserves the original first-candidate tie rule.
    return MU_VALUES[int(np.argmax(means))], means.tolist()


def evaluate(X_seen, y_seen, X_test, y_test, method, seed, n_init):
    mu, validation = select_mu(X_seen, y_seen, method, seed, n_init)
    model = build_model(method, seed, n_init, mu).fit(X_seen)
    return dict(method=method, mu=mu, validation_means=validation,
                train_acc=clustering_accuracy(y_seen, model.labels_),
                test_acc=clustering_accuracy(y_test, model.predict(X_test)))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_results(output, records, expected_splits):
    """Always summarize newly computed records, never the bundled example CSV."""
    fields = ("split", "seed", "method", "mu", "train_acc", "test_acc")
    with (output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        if not rows:
            continue
        item = dict(method=method, n=len(rows))
        for phase in ("train", "test"):
            values = np.array([r[f"{phase}_acc"] for r in rows]) * 100
            item[f"{phase}_mean_percent"] = float(values.mean())
            item[f"{phase}_std_percent"] = float(values.std(ddof=1)) if len(rows) > 1 else None
        summary.append(item)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "n", "train_mean_percent",
                                "train_std_percent", "test_mean_percent", "test_std_percent"))
        writer.writeheader()
        writer.writerows(summary)
    complete = len(records) == len(METHODS) * expected_splits
    lines = ["# MNIST results", "", "Status: " + ("complete" if complete else "partial"), "",
             "ACC in percent; sample SD across completed outer splits (N/A for one split).", "",
             "| Method | Splits | Train ACC | Test ACC |", "|---|---:|---:|---:|"]
    for item in summary:
        cells = []
        for phase in ("train", "test"):
            std = item[f"{phase}_std_percent"]
            cells.append(f"{item[f'{phase}_mean_percent']:.2f} ± " + (f"{std:.2f}" if std is not None else "N/A"))
        lines.append(f"| {item['method']} | {item['n']}/{expected_splits} | {' | '.join(cells)} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run(X, y, args, n_seen=3500):
    sources = [ROOT / "run_mnist.py", ROOT / "requirements.txt", *sorted((ROOT / "sec").glob("*.py"))]
    config = dict(dataset="mnist", data_sha256=SHA256, splits=args.splits, n_init=args.n_init,
                  seed=args.seed, threads=args.threads, seen=n_seen, total=len(X),
                  cv_folds=3, mu_values=list(MU_VALUES), python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    output = Path(args.output)
    if args.resume:
        saved = json.loads((output / "progress.json").read_text())
        if saved["config"] != config:
            raise ValueError("Resume requires identical configuration, source and environment")
        records = saved["records"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        records = []
        atomic_json(output / "progress.json", dict(config=config, records=records))
    completed = {(r["split"], r["method"]) for r in records}
    if len(completed) != len(records):
        raise ValueError("Duplicate saved results")
    for record in records:
        if (record["method"] not in METHODS or not 0 <= record["split"] < args.splits
                or record["seed"] != args.seed + record["split"]
                or any(not np.isfinite(record[k]) or not 0 <= record[k] <= 1 for k in ("train_acc", "test_acc"))):
            raise ValueError("Invalid saved result")
    with threadpool_limits(limits=args.threads):
        for split in range(args.splits):
            seed = args.seed + split
            order = np.random.default_rng(seed).permutation(len(X))
            train, test = order[:n_seen], order[n_seen:]
            X_seen, X_test = np.asarray(X[train], dtype=float), np.asarray(X[test], dtype=float)
            for method in METHODS:
                if (split, method) in completed:
                    continue
                print(f"Split {split + 1}/{args.splits}: {method}", flush=True)
                record = dict(split=split, seed=seed, **evaluate(
                    X_seen, y[train], X_test, y[test], method, seed, args.n_init))
                records.append(record)
                atomic_json(output / "progress.json", dict(config=config, records=records))
                write_results(output, records, args.splits)
                print(f"  train ACC={100 * record['train_acc']:.2f}%, test ACC={100 * record['test_acc']:.2f}%", flush=True)
    write_results(output, records, args.splits)
    print((output / "summary.md").read_text())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mnist")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.splits, args.n_init, args.threads) < 1:
        parser.error("splits, n-init and threads must be positive")
    if args.seed < 0 or args.seed + args.splits - 1 >= 2 ** 32:
        parser.error("all split seeds must be in [0, 2**32)")
    if args.output.exists() and not args.resume:
        parser.error("output exists; choose a fresh directory or use --resume")
    X, y = load_mnist(args.data_dir)
    run(X, y, args)


if __name__ == "__main__":
    main()
