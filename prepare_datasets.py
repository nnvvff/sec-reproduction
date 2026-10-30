"""Download documented public SEC benchmark sources; never synthesize missing datasets.

Outputs data/prepared/<name>.npz with row-sample X and one-dimensional y,
plus provenance metadata. The paper's exact preprocessing is not assumed.
"""
import argparse
import bz2
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file

ROOT = Path(__file__).resolve().parent
EXPECTED = {"optdigits": (5620, 64, 10), "usps": (9298, 256, 10),
            "mnist": (70000, 784, 10), "coil20": (1440, 16384, 20),
            "yaleb": (2414, 1024, 38), "pie": (3329, 1024, 68),
            "ar": (840, 768, 120), "mpeg7": (1400, 6000, 70)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, name, expected_hash=None):
    target = ROOT / "data" / "downloads" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"Downloading {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "SEC-Reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_bytes(content)
        temporary.replace(target)
    digest = sha256(target)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Checksum mismatch for {target}: {digest}")
    return target, {"url": url, "local_archive": str(target.relative_to(ROOT)), "sha256": digest}


def save(name, X, y, sources, processing, correspondence):
    X, y = np.asarray(X), np.asarray(y).reshape(-1)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"Expected a numeric 2D feature matrix for {name}")
    if not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError(f"Expected finite integer class labels for {name}")
    shape = (X.shape[0], X.shape[1], len(np.unique(y)))
    if X.ndim != 2 or len(y) != len(X) or shape != EXPECTED[name] or not np.isfinite(X).all():
        raise ValueError(f"Unexpected {name} shape: {shape}; expected {EXPECTED[name]}")
    destination = ROOT / "data" / "prepared"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{name}.npz"
    np.savez_compressed(path, X=X, y=y.astype(np.int64))
    counts = {str(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))}
    metadata = {"name": name, "shape": list(X.shape), "n_classes": shape[2],
                "dtype": str(X.dtype), "range": [float(X.min()), float(X.max())],
                "class_counts": counts, "sources": sources, "processing": processing,
                "paper_correspondence": correspondence, "npz_sha256": sha256(path),
                "prepared_date": "2026-10-29"}
    (destination / f"{name}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Prepared {path}: {shape}", flush=True)


def prepare_optdigits():
    from sec.data import load_optdigits
    X, y, metadata = load_optdigits(ROOT / "data" / "optdigits")
    save("optdigits", X, y, [metadata], "Concatenate UCI .tra then .tes, retain raw 0..16 values.",
         "Matches Table I sample count, dimension and classes. Exact author normalization is unspecified.")


def prepare_mnist():
    path, source = download("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", "mnist.npz",
                           "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1")
    source["primary_implementation"] = "https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py"
    with np.load(path, allow_pickle=False) as data:
        X = np.concatenate([data["x_train"], data["x_test"]]).reshape(70000, 784)
        y = np.concatenate([data["y_train"], data["y_test"]])
    save("mnist", X, y, [source], "Concatenate official 60k train then 10k test; C-order flatten 28x28, raw uint8 0..255.",
         "Full MNIST matches Table I. Keras distribution has pinned SHA-256. No resize or feature normalization.")


def prepare_usps():
    sources, arrays, labels = [], [], []
    for filename, md5 in [("usps.bz2", "ec16c51db3855ca6c91edd34d0e9b197"),
                          ("usps.t.bz2", "8ea070ee2aca1ac39742fdd1ef5ed118")]:
        mirror = "https://media.githubusercontent.com/media/Lallapallooza/cycle_adv_ot/e38f5b5389aee70ca1d26910214e6140a89df080/data/" + filename
        path, source = download(mirror, filename)
        if hashlib.md5(path.read_bytes()).hexdigest() != md5:
            raise ValueError(f"USPS official torchvision checksum mismatch: {filename}")
        source["official_url"] = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/" + filename
        source["verified_md5"] = md5
        source["checksum_source"] = "https://github.com/pytorch/vision/blob/18c1a42b3aa0b0429e609a2b245ef379a21be764/torchvision/datasets/usps.py"
        X, y = load_svmlight_file(io.BytesIO(bz2.decompress(path.read_bytes())), n_features=256, zero_based=False)
        arrays.append(X.toarray())
        labels.append(y.astype(int))
        sources.append(source)
    save("usps", np.concatenate(arrays), np.concatenate(labels), sources,
         "Concatenate LIBSVM 7291 train then 2007 test; preserve 256 values in [-1,1] and original 1..10 labels.",
         "Matches Table I size/dimension/classes. Both mirrored archives match official torchvision MD5 values. LIBSVM scaling is retained; author's exact input scaling is unspecified.")


def prepare_coil20():
    path, source = download("https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
                            "coil-20-proc.zip", "517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88")
    with zipfile.ZipFile(path) as archive:
        found = []
        for member in archive.namelist():
            match = re.search(r"obj(\d+)__(\d+)\.png$", member)
            if match:
                found.append((int(match[1]), int(match[2]), member))
        found.sort()
        if len(found) != 1440 or len({(a, b) for a, b, _ in found}) != 1440:
            raise ValueError("COIL archive does not contain all 1440 distinct poses")
        arrays = []
        for _, _, member in found:
            with Image.open(io.BytesIO(archive.read(member))) as im:
                pixels = np.asarray(im.convert("L"))
            if pixels.shape != (128, 128):
                raise ValueError(f"Unexpected COIL image size {pixels.shape}")
            arrays.append(pixels.reshape(-1))
    save("coil20", np.stack(arrays), [label for label, _, _ in found], [source],
         "Official processed images, object-major then pose order, grayscale raw uint8 0..255, C-order flatten 128x128. No resize.",
         "Matches Table I exactly in count, dimensions and classes. Author normalization was not specified.")


def prepare_yaleb():
    url = "https://raw.githubusercontent.com/jicongfan/Automated-Spectral-Clustering/c7656c25387003c5f80c18d51c71c94088970a26/YaleB_32x32.mat"
    path, source = download(url, "YaleB_32x32.mat", "3fbbda4eb781b9bb28025189677b6727706e114eb0215bbfd718f1f4fcb54b99")
    source["matching_mirror"] = "https://github.com/flatironinstitute/online_psp/blob/8acace43a7c13976dd4a13fd6629593088b2095a/datasets/YaleB_32x32.mat"
    source["git_blob_sha1"] = "cb46e1db16ee99cbe1af507149fb0b55ce9160c5"
    data = loadmat(path)
    save("yaleb", data["fea"], data["gnd"], [source],
         "Retain fea/gnd from the distributed YaleB_32x32.mat, raw uint8 0..255, no resizing or reordering.",
         "2414 x 1024 / 38 matches Table I. Identical blob in two independent research repositories. Not an authenticated SEC-author archive.")


def prepare_ar():
    path, source = download("https://raw.githubusercontent.com/csliangdu/RMKKM/5f0cd9a3ae1993eeaade7701b7085917595cde6d/data/AR_840n_768d_120c.mat",
                            "AR_840n_768d_120c.mat", "8a38c360629d06e227c44b4b60e8ffb24fb81563c97d805ab70818eec3f8ee23")
    source["research_implementation"] = "https://github.com/csliangdu/RMKKM"
    data = loadmat(path)
    save("ar", data["X"], data["y"], [source],
         "Retain raw uint8 X/y from RMKKM's unnormalized AR distribution. Archive records image height32, width24. No sample selection or resizing.",
         "840 x 768 / 120 matches Table I. Obtained from Liang Du's IJCAI2015 RMKKM research distribution; not an authenticated SEC-author archive.")


def prepare_pie():
    path, source = download("https://raw.githubusercontent.com/chamwen/JPDA/b056e5eca625fc480aa605242348809f9199d5f8/data/PIE27.mat",
                            "PIE27.mat", "85d801d89de60fa83d171b3029c1b5f52cb02a5927d43b23ada015db43acc30d")
    data = loadmat(path)
    save("pie", data["fea"], data["gnd"], [source],
         "Retain distributed float64 fea in [0,1] and gnd, all3329 frontal C27 faces and 1024 features. No resizing or sample selection.",
         "3329 x 1024 /68 matches Table I. JPDA IJCNN2020 author distribution. Pixel scaling/preprocessing is not authenticated against SEC inputs.")


def prepare_mpeg7():
    path, source = download("https://raw.githubusercontent.com/moetayuko/ExpCut_TAI/53911f8de853fe630b6a0dabb0a2bd8af9eacc48/data/Mpeg7_uni.mat",
                            "Mpeg7_uni.mat", "450a6f4ac811f8eb6c40192d69401441cf7ba3e661d7ccec3e2b0134173020f1")
    source["matching_feature_matrix"] = "https://github.com/huangdonghere/OBCut/blob/6409ffad2585213b8c90015bc613f7d09da968e7/data_Mpeg7.mat"
    data = loadmat(path)
    save("mpeg7", data["X"], data["Y"], [source],
         "Retain distributed X/Y exactly. Archive stores im_height60 and im_width100. No extraction, rescaling, resizing or sample selection.",
         "1400 x6000 /70 matches Table I. Official ExpCut2021 author repository (coauthored by Feiping Nie); X/Y exactly match independent OBCut2026 author distribution. SEC2011 pixel/descriptor equivalence is unverified.")


PREPARERS = {"optdigits": prepare_optdigits, "mnist": prepare_mnist, "usps": prepare_usps, "coil20": prepare_coil20, "yaleb": prepare_yaleb,
             "ar": prepare_ar, "pie": prepare_pie, "mpeg7": prepare_mpeg7}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=sorted(PREPARERS))
    args = parser.parse_args()
    for name in args.datasets:
        PREPARERS[name]()


if __name__ == "__main__":
    main()
