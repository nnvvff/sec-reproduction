# Spectral Embedded Clustering: MNIST Reproduction

Independent Python reproduction of Nie et al., **Spectral Embedded Clustering: A Framework for In-Sample and Out-of-Sample Spectral Clustering**, *IEEE Transactions on Neural Networks*, 2011. [Paper / DOI](https://doi.org/10.1109/TNN.2011.2162000).

## Repository structure

Core algorithms are in `sec/`; the root scripts prepare data and run experiments. Saved MNIST results and figures are included for inspection without retraining.

```text
sec/                   SEC implementation, baselines and utilities
prepare_datasets.py    Download and prepare data
run_benchmark.py       Run fixed baselines
run_cv_benchmark.py    Run SEC with training-internal CV
demo.py               Small offline example
tests/                Implementation and protocol checks
results/mnist/        Saved scores and experiment configurations
make_figures.py       Rebuild the result table and figures
figures/              Comparison plots
docs/                 Provenance, environment and verification
```

## 1. Reproduction code

The implementation covers **SEC/SC**, **SEC/LR**, spectral rotation and prediction for unseen samples. SEC combines a local structure term with global regression; the two variants use a graph Laplacian and local regression, respectively. KMeans and SC+ridge serve as baselines.

**Install and check** using Python 3.10–3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-data.txt
python demo.py
python -m pytest -q
```

The offline demo uses generated data. The retained **49 tests passed**; environment warnings and check details are recorded in [verification notes](docs/VERIFICATION.md).

**Reproduce the MNIST experiment:**

```bash
python prepare_datasets.py mnist
python run_benchmark.py --datasets mnist --methods KMeans SC+ridge \
  --threads 1 --output outputs/mnist_fixed
python run_cv_benchmark.py --datasets mnist --methods SEC/SC SEC/LR \
  --threads 1 --output outputs/mnist_cv
```

Defaults use **20 random splits, 50 initializations per fit and 3-fold inner CV**. Full runs are computationally substantial. To regenerate the figures from saved scores without training, run `python make_figures.py`. [Data preparation](DATASETS.md).

**Result-generating environment:** CPU execution, one numerical-library thread per fit; Python **3.10.12**, NumPy **2.1.3**, SciPy **1.15.3**, scikit-learn **1.6.1**, OpenBLAS **0.3.27**. Local verification used Python 3.13.5 on macOS. [Environment records](docs/EXPERIMENT_ENVIRONMENT.json).

## 2. Reproduction results

We present **one MNIST experiment**, using all 70,000 images as 784-dimensional raw pixel vectors. Each split contains **3,500 training and 66,500 test samples**, rather than the standard MNIST classification split. All methods share the same 20 outer partitions.

ACC below is **mean ± standard deviation (%)** across all 20 splits; our SD uses `ddof=1`. Paper values are from Tables II and III. SC+ridge corresponds to SC for training and SEC/SC (μ → 0) for testing.

| Method | Our train ACC | Paper train ACC | Our test ACC | Paper test ACC |
|---|---:|---:|---:|---:|
| KMeans | 52.99 ± 2.34 | 53.2 ± 2.3 | 52.71 ± 1.99 | 52.8 ± 1.8 |
| SC+ridge | 61.62 ± 3.23 | 61.7 ± 2.5 | 54.90 ± 2.20 | 55.1 ± 1.8 |
| SEC/SC | 59.00 ± 5.40 | 61.7 ± 2.5 | 54.32 ± 2.74 | 55.1 ± 1.8 |
| SEC/LR | 69.66 ± 5.09 | 71.0 ± 2.5 | 62.41 ± 3.74 | 62.9 ± 1.9 |

**The SEC test means are close to the paper:** SEC/SC is **0.78 percentage points** lower, and SEC/LR is **0.49 points** lower. The ordering also agrees: SEC/LR outperforms SEC/SC. Training means differ more, and our variability is larger; these results do not establish statistical equivalence.

![MNIST test accuracy compared with the paper](figures/mnist_test.png)

*Mean ± SD; error bars are not confidence intervals. [Training comparison](figures/mnist_train.png) · [Per-split scores](results/mnist/runs.csv) · [Summary](results/mnist/summary.csv).*

**Scope:** MNIST was selected from completed experiments as a teaching case with close test means; no random split was discarded. Inner validation labels select SEC parameters, while outer test data are excluded from selection. This is an explicit replacement CV protocol because the authors' full CV and preprocessing details were not recovered—not an exact or official reproduction. [Provenance](docs/PROVENANCE.md).

No software license has been selected; datasets and the paper PDF are not bundled.
