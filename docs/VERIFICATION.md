# Release verification

Checked on **2026-10-31**, without a full benchmark rerun or algorithm changes.

| Check | Result |
|---|---|
| Copied source/configuration/test files | All 21 byte-identical to the source project; SHA-256 manifest included |
| Core and runner provenance | All nine `sec` modules, both runners, `paper_protocol.py` and `requirements.txt` match the completed MNIST experiment's source hashes |
| Existing selected tests, existing local project environment (Python 3.13.5) | 49 passed; one warning from the explicit historical self-excluding LR neighborhood test |
| Fresh Python 3.13.5 environment | `pip install -r requirements-data.txt` succeeded; `pip check` found no broken requirements |
| Existing selected tests, fresh environment | 49 passed; NumPy matrix-operation runtime warnings were also emitted (472 warnings in the recorded repeat) |
| Offline demo, both local environments | SEC/SC and SEC/LR each reported train/test ACC 1.000 on generated data; finite eigen residuals around 1e-16 |
| Additional fresh-environment finite-value check | All fitted numeric arrays, eigen residuals and predictions were finite for both variants on a 300-sample generated dataset |
| MNIST download/preparation | Command succeeded; X/y arrays exactly equal the original experiment's prepared input |
| Published per-split records | All 80 published MNIST source records matched the original audit's stored metadata hashes and exported ACC fields |
| Main presentation metrics | Train/test ACC recomputed from original saved predictions for all 80 fits; agreement within 1e-14 |
| Table and figures | Regenerated from all 80 scores; mean/sample SD agree with the original summary within 1e-12 percentage points; both PNG figures visually inspected |
| Runner interfaces | Both documented benchmark entry points load and `--help` succeeds |
| Release contents | Relative Markdown links checked; data, environments, models and archives ignored; intended files scanned for machine paths and credential patterns |

The fresh environment's NumPy operations emitted divide-by-zero, overflow and invalid-operation warnings even while the numerical tests passed and the additional fitted-array checks were finite. Their cause was not established in this packaging task. They were not suppressed in the implementation, and the checks do not certify arbitrary datasets or platforms as warning-free. The original experiment used its recorded Python 3.10.12 environment; the new installation was not used to replace its benchmark scores. Exact fresh-install package versions are in [`VALIDATION_ENVIRONMENT.txt`](VALIDATION_ENVIRONMENT.txt).

The computer's default `python3` was Python 3.14.4. An initial installation with it was canceled; the README therefore explicitly chooses Python 3.13 and states the supported dependency range. No dependency versions in the original `requirements.txt` were changed.

The original project's 139-test report and 28,660 completed fits are historical evidence. This release ran the retained **49-test subset**, small generated-data checks, data preparation and figure generation only. It did not re-execute the full independent model audit, repeat benchmark training, or check other dataset downloads.

The final MNIST figures compare the complete 20-split case study with paper Tables II/III. Paper mean/SD values were checked directly in the supplied paper; the test table was also visually inspected. Experiment versions come from MNIST run configurations, while OpenBLAS/CPU capacity come from the full-run resource report. The preceding OS record is explicitly dated and is not represented as an OS snapshot captured during the MNIST run.
