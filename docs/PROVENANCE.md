# MNIST result provenance

This release presents only the MNIST case study from the completed `full_revision_20261030` experiment: KMeans, SC+ridge, SEC/SC and SEC/LR, with all 20 outer splits per method. MNIST was selected after the broader experiment as an illustrative case with test means close to the paper, not as evidence of uniform replication across datasets. No new benchmark training was performed for this release.

[`RESULT_PROVENANCE.json`](RESULT_PROVENANCE.json) records the original summary/audit hashes. The historical full-run audit reports `passed=true`; its wider fit/model counts are historical context, not this repository's experiment size or newly executed checks. The published 80 result records were matched to that audit's stored metadata hashes, and their train/test ACC fields were matched to the original CSV. Train/test ACC were also independently recomputed from the 80 original prediction files during packaging. The large source archives and models are retained outside this compact repository.

[`UNCHANGED_SOURCE_SHA256.json`](UNCHANGED_SOURCE_SHA256.json) lists 21 copied source, configuration and test files, all byte-identical to the original project. All nine `sec/*.py` modules, both benchmark runners, `paper_protocol.py` and `requirements.txt` also match the completed MNIST run's recorded source hashes. New presentation code produces the table and figures; the algorithm is unchanged.

The original `results/mnist/cv_config.json` describes the first five-split shard; the other three shards use the same protocol with disjoint split indices. The original fixed configuration includes KM-r as well as the two displayed baselines, and the original CV configuration includes other methods. They are retained as provenance records; only the four README methods appear in the published scores. The configuration's source hashes refer to runtime code, not the new presentation files.

The `result_path` CSV field is a locator relative to the original run root; the raw result JSON and models are not bundled. The `candidate` field records selected CV settings, and is blank for fixed baselines. Per-split ACC uses fractions in [0, 1]; summaries and figures use percentages. SD is calculated across all 20 outer splits with `ddof=1`.

`paper_reference.json` contains the MNIST mean/SD values transcribed and checked against Tables II and III of the 2011 paper. SC+ridge corresponds to SC in Table II and SEC/SC (μ → 0) in Table III. No paper pages or PDF files are redistributed. Exact author data, preprocessing and CV details remain unrecovered.

The original project and experiment records are preserved. The current repository presentation contains no other dataset results, parameter sweeps or Nyström comparison. Generic supporting source options remain unchanged. No software license has been selected.
