"""Download and load the full MNIST archive used by the experiment."""
import hashlib
from pathlib import Path
import urllib.request
import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def load_mnist(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "mnist.npz"
    if not archive.exists():
        print("Downloading MNIST...", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("MNIST download checksum mismatch")
        temporary = archive.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("MNIST archive checksum mismatch")
    with np.load(archive, allow_pickle=False) as data:
        X = np.concatenate([data["x_train"], data["x_test"]]).reshape(70000, 784)
        y = np.concatenate([data["y_train"], data["y_test"]])
    if X.shape != (70000, 784) or y.shape != (70000,) or len(np.unique(y)) != 10:
        raise ValueError("Unexpected MNIST dimensions or class count")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite MNIST data")
    return X, y
