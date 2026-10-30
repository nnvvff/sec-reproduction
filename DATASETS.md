# Data sources

Run `python prepare_datasets.py optdigits` for the main example. The downloader writes `data/prepared/<name>.npz` with row-sample `X` and integer `y`, plus a JSON record of source URLs, processing and checksums. Data and downloaded archives are ignored by Git. `requirements-data.txt` includes Pillow because the preparation module imports it for image datasets.

## Main example

[UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak (1998), [dataset DOI](https://doi.org/10.24432/C50P49). The UCI page lists CC BY 4.0. The loader uses the 64-feature files, concatenating 3,823 training rows and 1,797 test rows in that order. Values stay in 0–16, and labels are 0–9. It does not use the 1,797-row `sklearn.load_digits()` subset. SEC evaluation uses fresh randomized splits of all 5,620 rows.

## Supplement sources

The existing experiment used the following inputs. Sizes match the paper's Table I but do not authenticate the authors' original matrices or preprocessing. The preparation script records additional source metadata and enforces the declared shapes; consult the original providers' terms before downloading or redistributing.

| Key | Samples × features | Classes | Seen fraction | Source and conversion |
|---|---:|---:|---:|---|
| ar | 840 × 768 | 120 | 0.60 | [RMKKM](https://github.com/csliangdu/RMKKM/tree/5f0cd9a3ae1993eeaade7701b7085917595cde6d), `data/AR_840n_768d_120c.mat`; retain X/y |
| yaleb | 2414 × 1024 | 38 | 0.60 | [Automated Spectral Clustering](https://github.com/jicongfan/Automated-Spectral-Clustering/tree/c7656c25387003c5f80c18d51c71c94088970a26), `YaleB_32x32.mat`; retain fea/gnd |
| pie | 3329 × 1024 | 68 | 0.60 | [JPDA](https://github.com/chamwen/JPDA/tree/b056e5eca625fc480aa605242348809f9199d5f8), `data/PIE27.mat`; retain distributed floating-point features |
| mpeg7 | 1400 × 6000 | 70 | 0.60 | [ExpCut_TAI](https://github.com/moetayuko/ExpCut_TAI/tree/53911f8de853fe630b6a0dabb0a2bd8af9eacc48), `data/Mpeg7_uni.mat`; retain distributed X/Y |
| coil20 | 1440 × 16384 | 20 | 0.60 | [Columbia CAVE](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php), processed 128×128 grayscale images; sorted by object and angle, C-order flatten |
| optdigits | 5620 × 64 | 10 | 0.60 | UCI, as above |
| usps | 9298 × 256 | 10 | 0.20 | [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps), distributed train then test, values −1 to 1 |
| mnist | 70000 × 784 | 10 | 0.05 | [Keras](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py), 60,000 train then 10,000 test; raw pixels, C-order flatten |

USPS downloads use a [pinned research mirror](https://github.com/Lallapallooza/cycle_adv_ot/tree/e38f5b5389aee70ca1d26910214e6140a89df080/data) and verify the two original archive MD5 values against torchvision's published checksums. MNIST verifies the Keras archive SHA-256. The script contains the exact pinned mirror URLs for the other distributed matrices. These mirrors are provenance for the chosen inputs, not evidence of equivalence to SEC's 2011 inputs.

Other dataset keys can be passed to `prepare_datasets.py` explicitly. Only the Optdigits download is exercised during this release; the supplementary sources describe the previous experiment.
