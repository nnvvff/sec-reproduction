"""Full UCI Optdigits (5620), not sklearn.load_digits() (1797)."""

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip"


def load_optdigits(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "optdigits-uci.zip"
    if not archive.exists():
        # Download only data; no archive extraction or execution.
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            if not {"optdigits.tra", "optdigits.tes"}.issubset(z.namelist()):
                raise ValueError("Unexpected UCI archive")
        archive.write_bytes(content)
    with zipfile.ZipFile(archive) as z:
        arrays = [np.loadtxt(io.BytesIO(z.read(name)), delimiter=",")
                  for name in ("optdigits.tra", "optdigits.tes")]
    data = np.concatenate(arrays)
    if data.shape != (5620, 65) or not np.isfinite(data).all():
        raise ValueError(f"Expected full Optdigits shape (5620, 65), got {data.shape}")
    metadata = {"url": URL, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "shape": list(data.shape), "order": ["optdigits.tra", "optdigits.tes"],
                "preprocessing": "raw 0..16 features; train-only centering inside SEC"}
    (directory / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return data[:, :-1], data[:, -1].astype(int), metadata
