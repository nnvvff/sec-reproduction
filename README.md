# Spectral Embedded Clustering: MNIST Reproduction

Independent Python reproduction of Nie et al., **Spectral Embedded Clustering: A Framework for In-Sample and Out-of-Sample Spectral Clustering**, *IEEE Transactions on Neural Networks*, 2011. [Paper / DOI](https://doi.org/10.1109/TNN.2011.2162000).

## 1. File structure

```text
run_mnist.py       Single experiment entry point and result summary
sec/               SEC algorithms, CV cache, MNIST loader and ACC
requirements.txt   Four runtime dependencies
results/mnist/     Saved 20-split results (not overwritten by new runs)
outputs/           Your new results and resume state (created at runtime)
```

## 2. Installation and running

The experiment runs **SEC/SC and SEC/LR**, the two SEC variants reproduced here. SEC combines local structure with global regression to cluster training samples and predict clusters for unseen samples.

Use Python **3.10–3.13**. Dependencies are pinned in `requirements.txt`.

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

If needed, replace `python3` with a supported interpreter such as `python3.13`. On Windows Command Prompt, activate the environment with `.venv\Scripts\activate`.

**Run the full experiment:**

```bash
python run_mnist.py
```

**Optional check:** run one split with fewer initializations. This uses the full dataset but does not reproduce the 20-split results reported below.

```bash
python run_mnist.py --splits 1 --n-init 3 --output outputs/check
```

## 3. Output

The script automatically downloads MNIST and verifies its SHA-256. By default, it runs **20 random splits with 50 initializations per fit** and saves results to `outputs/mnist/`:

- `runs.csv`: train/test ACC as fractions in [0, 1] and selected μ for each split and method.
- `summary.csv` and `summary.md`: mean ± sample standard deviation, in percent.
- `progress.json`: settings, completed results and CV scores for resuming.

Resume an interrupted run with `python run_mnist.py --resume`. Completed results are reused; an interrupted method is rerun. For custom runs, keep the same arguments and add `--resume`. Use a new `--output` directory to start a fresh run.

**Protocol:** concatenate all 70,000 images, retain raw 0–255 pixels, and use 3,500 training / 66,500 test samples per split. Seeds are `20261029 + split`. SEC selects μ from `1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9` by three-fold CV using training-internal validation labels; outer test data do not enter selection. Initializations are selected by optimization objective. SC uses a union 5-neighbor graph and seventh-neighbor scale; LR uses the center plus four neighbors.

## 4. Results and paper comparison

Results are reported as mean ± standard deviation (%) over 20 random splits. Paper results are taken from Tables II and III.

| Method | Our train ACC | Paper train ACC | Our test ACC | Paper test ACC |
|---|---:|---:|---:|---:|
| SEC/SC | 59.00 ± 5.40 | 61.7 ± 2.5 | 54.32 ± 2.74 | 55.1 ± 1.8 |
| SEC/LR | 69.66 ± 5.09 | 71.0 ± 2.5 | 62.41 ± 3.74 | 62.9 ± 1.9 |

On MNIST, the same dataset used in the paper, our mean test accuracies are **0.78 percentage points** lower for SEC/SC and **0.49 points** lower for SEC/LR. These close test results provide additional evidence supporting the correctness of our reproduction.

[Saved per-split scores](results/mnist/runs.csv) · [Saved summary](results/mnist/summary.csv).

## 5. Experiment environment

**Environment for the saved results:** CPU execution, one numerical-library thread per fit; Python **3.10.12**, NumPy **2.1.3**, SciPy **1.15.3**, scikit-learn **1.6.1**, OpenBLAS **0.3.27**.
