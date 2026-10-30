"""Atomic experiment artifacts and strict resume validation."""
import hashlib
import json
from pathlib import Path
import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def ensure_split(path, **expected):
    path = Path(path)
    if not path.exists():
        atomic_npz(path, **expected)
        return
    try:
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(expected):
                raise ValueError("Unexpected split fields")
            for key, array in expected.items():
                if not np.array_equal(saved[key], array):
                    raise ValueError(f"Split field {key} differs from deterministic protocol")
    except Exception as error:
        raise ValueError(f"Invalid split checkpoint {path}: {error}") from error


def validate_completed_fit(directory, expected, n_train, n_test):
    """Return False for uncommitted work; corrupt committed work raises."""
    directory = Path(directory)
    result_path = directory / "result.json"
    if not result_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text())
        for key, value in expected.items():
            if result[key] != value:
                raise ValueError(f"Result identity mismatch: {key}")
        with np.load(directory / "predictions.npz", allow_pickle=False) as pred:
            if pred["train"].shape != (n_train,) or pred["test"].shape != (n_test,):
                raise ValueError("Prediction length mismatch")
            if not np.isfinite(pred["train"]).all() or not np.isfinite(pred["test"]).all():
                raise ValueError("Nonfinite predictions")
        with np.load(directory / "model.npz", allow_pickle=False) as model:
            if not model.files:
                raise ValueError("Empty saved model")
            for key in model.files:
                value = model[key]  # Force CRC / decompression checks on every member.
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite saved model field {key}")
        for filename, digest in result.get("artifact_sha256", {}).items():
            if filename not in ("model.npz", "predictions.npz"):
                raise ValueError("Unexpected checkpoint artifact name")
            if file_sha256(directory / filename) != digest:
                raise ValueError(f"Checksum mismatch: {filename}")
    except Exception as error:
        raise ValueError(f"Invalid completed checkpoint {directory}: {error}") from error
    return True
