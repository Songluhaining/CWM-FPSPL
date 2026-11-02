# Required Inputs and Where to Put Them

## External data (not included in this starter pack)
1. `third_party/CLAP/` — clone https://github.com/ttrangnguyen/CLAP
2. `third_party/PULP/` — clone https://github.com/dsjads/PULP
3. `data/raw/spl_benchmark_v2/` — download the SPLC2021 benchmark V2 from https://tuanngokien.github.io/splc2021/ . Do not confuse 1,570-version benchmark V2 with the CLAP/PULP 823-version FP-detection subset.
4. If available, `third_party/existing_instrumenter/` — place the user's pre-existing Java instrumentation/LDDG builder here, or point the config to an external copy. Never edit this without first making a separate branch/copy.
5. Papers already supplied in `references/`.

## Confirmed benchmark artifact names (do not infer other names)
From SPL benchmark: `model.m`, `model.m.ca4.csv`, `configs/`, `features/`, `variants/<product>/src/`, `variants/<product>/test/`, `variants/<product>/build/`, `variants/<product>/coverage/passed/`, `variants/<product>/coverage/failed/`, `spectrum_passed_coverage.xml`, `spectrum_failed_coverage.xml`, `roles.meta`, `batch.test.*.flag`, `config.report.csv.done`, and `X.mutant.log`.
From PULP README: `variant_labels.csv` (F/FP/TP), `attributes.csv` (product features), `classified_testing_v1301.csv` (predictions); actual availability must be checked.

## Expected baseline commands and review
- PULP entry `PULP.py`, label generation `from consistent_testing_manager.LabelData import label_data`; `label_data(system_paths)` outputs `variant_labels.csv` per version according to repo README.
- `calculate_attributes_from_system_paths(system_paths)` creates `attributes.csv`.
- PULP source includes Python 3.7-era code and Java plugins. Use an isolated reproduction environment before changing legacy source.
- CLAP `CLAP.py` includes `system_based_classification`, `version_based_classification`, `product_based_classification`, `within_system_classification`.

## Data audit must answer BEFORE graph/model work
- Count exact system/version/product/test instances and compare with benchmark paper: 823 buggy versions, 10,433 failing, 14,191 FP, 22,555 TP (numbers for the benchmark paper's subset, not a promised count for your downloaded subset).
- Map every `(system_id, bug_version_id, product_id)` to its original test result, label source, feature configuration, and source folder. Normalize aliases (`Elevator-FH-JML`, `Email-FH-JML`).
- Determine origin `natural` vs `simulated` from verifiable provenance. If not available, set `unknown` and report—never infer by label alone.
- Identify how missing tests/removed failure evidence were generated; exclude hidden original failing tests from any online evaluation or intervention candidate pool.
- Inspect dynamic logs, if any: events? variable values? def-use edges? branch decisions? test inputs? statement source mapping? If unavailable set corresponding capabilities false.
- Label generation accesses true fault info only in an isolated stage. Prevent derived label features (bug line, patch, failure deletion provenance) from leaking into models.

## Preferred practical order
A. Start with BankAccountTP small smoke subset (one buggy version, 2-3 products); then Elevator; scale to the other four systems.
B. Reuse test coverage and mapping code as read-only input.
C. Implement Java 8 instrumentation only for missing actual dynamic events; keep a test-runner timeout and bounded event buffer.
D. For high-volume graphs use JSONL for debugging and parquet/PT for training after a validated schema migration.
