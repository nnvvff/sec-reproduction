# Result provenance and release scope

This release packages the completed `full_revision_20261030` experiment. It does not run new benchmark training or combine that experiment with older fixed-parameter sweeps.

The original summary reports 26,640 inner-CV fits, 800 selected final fits, 480 fixed baseline fits and 740 Nyström candidate fits: **28,660 total**. Its stored independent audit reports `passed=true`, no errors, 28,800 model-file hashes checked (including 140 derived winner copies), 1,420 final/selected predictions reconstructed, and 120 inner-model prediction checks. These are **historical audit results**, not checks rerun for this release. Likewise, the original project report records 139 passing tests; this compact release retains and runs a smaller relevant subset.

[`RESULT_PROVENANCE.json`](RESULT_PROVENANCE.json) records original summary/audit hashes and counts. During packaging, all 1,280 published main-method records were matched to their historical audited metadata hashes and their train/test ACC fields were matched to the exported CSV. The main presentation uses 80 of these records, with all 20 splits for each of four methods. The original audit and large model/prediction archives remain outside this compact repository.

[`UNCHANGED_SOURCE_SHA256.json`](UNCHANGED_SOURCE_SHA256.json) lists the 21 copied source, configuration and test files. They are byte-identical to the source project. All nine `sec/*.py` modules, both benchmark runners, `paper_protocol.py`, and `requirements.txt` also match the source hashes saved by the completed Optdigits CV experiment. There were no algorithm changes. New files provide publication packaging, documentation, figure generation and an explicit Pillow dependency for data preparation.

The two configuration files in `results/optdigits/` are original records: `cv_config.json` describes the first five-split shard of the 20-split run; the other shards use the same configuration with disjoint split indices. `fixed_config.json` includes the original three fixed baselines, although the main presentation displays only KMeans and SC+ridge. Their hashes refer to the historical runtime sources, not the newly added presentation files.

`result_path` in the published CSVs is a provenance locator relative to the original run root, not a claim that the raw result JSON or model is bundled. `candidate` records the selected CV settings; blank entries belong to fixed baselines. ACC in per-split CSVs is a fraction in [0, 1]; summaries and figures convert it to percent.

Nyström rows are deliberately excluded from both tables. In the original run, AR, Yale-B, PIE, MPEG7 and COIL20 used test-label bandwidth selection (oracle); Optdigits and USPS had a fixed bandwidth. The retained Nyström module supports existing baseline tests but is not part of the main experiment commands.

The source project, data, model archives and original experiment records are preserved outside the release. This repository excludes orchestration, remote-machine information, audit machinery, historical snapshots, downloaded datasets and the paper PDF. It has no selected software license.
