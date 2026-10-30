# Eight-dataset supplement

[`summary.csv`](summary.csv) contains 64 rows: eight methods on eight datasets, each over all 20 outer splits. [`runs.csv`](runs.csv) contains all 1,280 corresponding per-split scores. These are the same 2026-10-30 experiment as the main Optdigits presentation.

KMeans, SC+ridge and KM-r use fixed settings. DKM, LL, CLGR, SEC/SC and SEC/LR use training-internal 3-fold CV. LL/CLGR select among linear, cosine and RBF kernels; DKM/CLGR/SEC parameters use the nine-point grid in `paper_protocol.py` (seven points through 1e9 for MNIST). Preprocessing is raw. All methods use 50 initializations, and no outer splits are discarded. See the main README for graph conventions and label use.

Summary mean/SD fields are percentages; SD is the sample standard deviation (`ddof=1`). `paper_*` columns contain transcribed reference means, not this implementation's measurements. `nonfull_clusters` counts fits with fewer nonempty clusters than requested; these outcomes are retained. Per-split ACC fields are fractions. Nyström and historical fixed-parameter sweeps are not included.

Results on Yale-B and Optdigits still differ substantially from the paper. Other close values do not establish exact protocol recovery. Data provenance and preprocessing limitations apply to every dataset.

## Held-out ACC (%)

Mean ± sample SD over 20 splits. Full eight-method train/test results are in the CSV.

| Dataset | KMeans | SC+ridge | SEC/SC | SEC/LR |
|---|---:|---:|---:|---:|
| ar | 34.87 ± 1.20 | 42.68 ± 1.98 | 64.23 ± 2.55 | 63.59 ± 2.23 |
| yaleb | 12.17 ± 0.52 | 29.45 ± 1.10 | 31.79 ± 1.58 | 31.25 ± 1.21 |
| pie | 15.55 ± 0.59 | 32.32 ± 1.29 | 65.64 ± 1.78 | 65.74 ± 1.30 |
| mpeg7 | 48.37 ± 1.85 | 55.52 ± 2.04 | 55.45 ± 2.37 | 55.40 ± 1.81 |
| coil20 | 67.40 ± 3.16 | 77.41 ± 3.60 | 78.08 ± 3.13 | 76.94 ± 4.19 |
| optdigits | 79.42 ± 0.79 | 83.36 ± 4.42 | 83.32 ± 4.42 | 84.80 ± 3.81 |
| usps | 65.83 ± 2.04 | 60.66 ± 2.26 | 68.87 ± 0.52 | 73.69 ± 2.61 |
| mnist | 52.71 ± 1.99 | 54.90 ± 2.20 | 54.32 ± 2.74 | 62.41 ± 3.74 |
