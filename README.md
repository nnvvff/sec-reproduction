# Spectral Embedded Clustering: MNIST Reproduction

## 1. File structure

```text
run_mnist.py       Single experiment entry point and result summary
sec/               SEC algorithms, CV cache, MNIST loader and ACC
requirements.txt   Four runtime dependencies
results/mnist/     Saved 20-split results (not overwritten by new runs)
outputs/           Your new results and resume state (created at runtime)
```

## 2. Installation and running

Independent Python reproduction of Nie et al., **Spectral Embedded Clustering: A Framework for In-Sample and Out-of-Sample Spectral Clustering**, *IEEE Transactions on Neural Networks*, 2011. [Paper / DOI](https://doi.org/10.1109/TNN.2011.2162000).

The experiment runs **SEC/SC and SEC/LR**, the two SEC variants reproduced here. SEC combines local structure with global regression to cluster training samples and predict clusters for unseen samples.

Use Python **3.10–3.13** (not 3.14). The only runtime dependencies are NumPy, SciPy, scikit-learn and threadpoolctl, pinned in `requirements.txt`; no GPU, plotting package or test framework is required.

**Download and install** (macOS/Linux shell):

```bash
git clone https://github.com/nnvvff/sec-reproduction.git
cd sec-reproduction
python3 --version  # must be 3.10–3.13
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

The repository is public; cloning does not require a GitHub account. Alternatively, download and extract its ZIP, then open a terminal in the extracted folder. Check the version before creating the environment. If `python3` is outside 3.10–3.13, use a supported interpreter (for example `python3.13`) in both commands. On Windows, use `.venv\Scripts\activate` instead of `source`.

**Check the complete workflow first** with one split and fewer initializations:

```bash
python run_mnist.py --splits 1 --n-init 3 --output outputs/check
```

This still uses the full MNIST dataset and performs 44 fits, so allow time for CPU computation. Success means the process exits normally and `outputs/check/summary.md` reports `Status: complete` with two SEC variants and `1/1` splits. SD is `N/A` for one split. These are verification results, not the published 20-split results.

**Run the full experiment:**

```bash
python run_mnist.py
```

## 3. Output

With the default full-run command, the script downloads the [Keras-distributed MNIST archive](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py), verifies its SHA-256, and runs **20 random splits with 50 initializations per fit**. It writes **newly computed results** to `outputs/mnist/`:

- `runs.csv`: train/test ACC as fractions in [0, 1] and selected μ for each split and method.
- `summary.csv` and `summary.md`: mean ± sample SD, also printed when complete.
- `progress.json`: settings, completed results and CV scores for resuming.

A full run performs 880 fits, including inner CV, and can take substantial CPU time. Its final summary contains two SEC variants with `20/20` splits and ACC in percent. Resume completed methods with `python run_mnist.py --resume`; an interrupted method is rerun. Use `--output outputs/another_run` for a fresh run. To resume the short check, repeat its exact command with `--resume`. Existing output directories are never silently overwritten.

**Protocol:** concatenate all 70,000 images, retain raw 0–255 pixels, and use 3,500 training / 66,500 test samples per split. Seeds are `20261029 + split`. SEC selects μ from `1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9` by three-fold CV using training-internal validation labels; outer test data do not enter selection. Initializations are selected by optimization objective. SC uses a union 5-neighbor graph and seventh-neighbor scale; LR uses the center plus four neighbors.

## 4. Results and paper comparison

The bundled results were generated on **2026-10-30**, before this entry-point simplification. Entries are **mean ± standard deviation (%)** over all 20 random splits; our SD uses `ddof=1`. Paper references are from Tables II and III.

| Method | Our train ACC | Paper train ACC | Our test ACC | Paper test ACC |
|---|---:|---:|---:|---:|
| SEC/SC | 59.00 ± 5.40 | 61.7 ± 2.5 | 54.32 ± 2.74 | 55.1 ± 1.8 |
| SEC/LR | 69.66 ± 5.09 | 71.0 ± 2.5 | 62.41 ± 3.74 | 62.9 ± 1.9 |

**The SEC test means are close to the paper:** SEC/SC is **0.78 percentage points** lower and SEC/LR is **0.49 points** lower. SEC/LR has the highest test ACC in both comparisons. Training means differ more and our variability is larger. This is numerical agreement, not statistical equivalence.

[Saved per-split scores](results/mnist/runs.csv) · [Saved summary](results/mnist/summary.csv).

We conducted experiments on MNIST, the same dataset used in the paper, and obtained similar mean test accuracies. This agreement provides additional evidence supporting the correctness of our reproduction.

## 5. Experiment environment

**Environment for the saved results:** CPU execution, one numerical-library thread per fit; Python **3.10.12**, NumPy **2.1.3**, SciPy **1.15.3**, scikit-learn **1.6.1**, OpenBLAS **0.3.27**. Local refactoring checks used Python 3.13.5 on macOS. Floating-point results can vary across platforms; NumPy runtime warnings observed in the local environment did not prevent the checked numerical comparisons from passing.
