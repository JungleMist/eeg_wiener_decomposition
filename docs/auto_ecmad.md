# EEGNet-guided automatic ECMAD

`eeg_bg.auto_ecmad.AutoECMADPipeline` implements independent directed-reference
selection on ERP-CORE ERN. For each target/reference candidate, a fresh EEGNet
receives **only two channels**, `[coherent, specific]`. The lowest internal
selection subject-equal class-weighted BCE determines each target's reference.
The final freshly initialized EEGNet receives all selected pairs in target
order, `[target1::coherent, target1::specific, ...]`.

Each reference is selected independently. This does not guarantee a jointly
optimal channel combination or establish physiological denoising quality.
With 30 channels the full run fits 870 candidate models, a selected 60-channel
model and a 30-channel raw baseline. Defaults use one seed/split, not repeated
cross-validation.

## Run and resume

Run from the project root using the `eeg_pipeline` environment:

```bash
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py all \
  --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
```

The stages can also be executed separately. Use identical config and overrides
on every invocation:

```bash
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py prepare --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py search --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py train --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py evaluate --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
```

Omitting `--config` selects `configs/erp_core_auto_ecmad_eegnet.yaml`, the full
30-channel configuration (872 fits). Optional overrides are
`--data-dir`, `--output-dir`, `--device`, `--workers`, and `--force`.
Workers parallelize raw preparation only; fits are serial. Pin a physical CUDA
GPU externally with `CUDA_VISIBLE_DEVICES=N` if needed.

Source content hashes, config, package/backend code, dependency versions, trial grids and training
settings identify a run. Subsequent stages check source inventory and file
metadata; used prepared caches and completed model artifacts are checksummed.
After editing data or settings, run `all --force` or use a new output directory.
An interrupted candidate is retrained from scratch; completed candidates resume.
Do not run concurrent writers against one output directory.

For a real-data smoke run, create `configs/local_auto_ecmad_smoke.yaml` extending
the default config, set `auto_ecmad.adapter.subjects` to at least five available
subjects, `channels` to three EEG channels, and `training.max_epochs` to 1–2.
Use a separate `paths.results_dir`, CPU if desired, and run `all`. This checks
the workflow, not performance. The full experiment is intentionally expensive.

## Isolation and loss

Splits are by subject: 20% test; 20% of the remainder validation; 20% of the
remaining training subjects internal selection. Each holdout count rounds up,
preserving a nonempty fitting partition. Seed defaults to 42. First-stage
training receives internal-fit and internal-selection trials only. Search does
not load external validation/test EEG caches. Final training uses both internal
partitions, and external validation chooses AUPRC checkpoint and balanced
accuracy threshold. Both final models must be frozen before test loading.

Raw EEG alone fixes the rejection mask, events and trial IDs. The loss first
averages class-weighted trial BCE within each subject and then across subjects.
`w0=1`, `w1=N0/N1`, where counts come only from the fitting partition.
Selection uses the same fixed class weights. Per-trial normalization is applied
after baseline correction; reconstruction is checked **before** normalization.

ECMAD uses the existing continuous 50% overlap-add implementation, 12-second
windows, phasegated mode at 1 rad, no power gate, no protected band, and zero
coherence admission threshold. Numerical stability safeguards remain active.
Output includes status counts for the target direction and rejected raw trials.
Constant-signal pairs and recordings too short to resolve a scoring-band bin
pass through with zero coherent output and an explicit diagnostic reason.

## Unattended training and post-run analysis

`scripts/run_auto_ecmad_training_job.py` supervises the existing full pipeline
and invokes `scripts/analyze_auto_ecmad_results.py` only after successful
training/evaluation. `job_status.json` records supervisor/child PIDs, the exact
command, fixed device and phase. An advisory lock prevents duplicate supervisor
jobs writing the same output directory. Signals and nonzero exits produce an
explicit interrupted/failed status. Use a separate output directory for probes.

For a terminal launch on macOS, select a verified device and keep the job awake:

```bash
conda run --no-capture-output -n eeg_pipeline caffeinate -i python \
  scripts/run_auto_ecmad_training_job.py \
  --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml \
  --output-dir results/auto_ecmad_ern_3ch --device mps
```

Do not launch this again while `job_status.json` describes an active job. The
2026-10-21 full run was stopped at the user's request after 43 completed
candidates; its thread follow-up was deleted. Those partial artifacts remain in
`results/auto_ecmad_ern` and are not a complete training result. The CPU device
probe was archived separately under `results/auto_ecmad_ern_cpu_probe_20261021`.

The replacement end-to-end test uses
`configs/erp_core_auto_ecmad_eegnet_3ch.yaml`: FP1/F3/C3, all eligible subjects,
the same split and raw-derived rejection mask, and the original EEGNet/200-epoch
maximum training budget. It fits six directed two-channel candidates, a
six-channel selected model and a three-channel Raw baseline. Its artifacts live
under `results/auto_ecmad_ern_3ch`; this is distinct from the earlier five-subject,
one-epoch smoke test. `scope_verification.json` records the input/split checks.

Post-run analysis can also be repeated without retraining:

```bash
conda run -n eeg_pipeline python scripts/analyze_auto_ecmad_results.py \
  --results-dir results/auto_ecmad_ern_3ch
```

Analysis refuses incomplete candidate searches or inconsistent mappings,
checksums, trial identities and summary metrics. It exports a Chinese Markdown
report, self-contained HTML, six scientific figures in PNG/PDF, channel-reference
rankings, candidate training lengths, calibration, per-subject metrics, paired
test error transitions and subject-cluster bootstrap intervals (1000 repetitions
by default). Bootstrap intervals are conditional on frozen models and this test
split; they do not estimate retraining/reference-selection uncertainty. Reports
from smoke runs must not be interpreted as full training results.

## Python interfaces and inference

```python
from eeg_bg.auto_ecmad import AutoECMADPipeline

pipeline = AutoECMADPipeline("configs/erp_core_auto_ecmad_eegnet_3ch.yaml")
pipeline.run()

inference = AutoECMADPipeline.from_bundle(
    "results/auto_ecmad_ern_3ch/bundle", data_dir="/path/to/new/ERP_CORE", device="cpu",
    output_dir="results/auto_ecmad_new_predictions",
)
predictions = inference.predict()  # labels are not needed
```

Standalone CLI inference:

```bash
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py predict \
  --bundle results/auto_ecmad_ern_3ch/bundle --data-dir /path/to/new/ERP_CORE \
  --device cpu --output-dir results/auto_ecmad_new_predictions
```

The ERP adapter needs stimulus/response annotations to locate response windows.
It does not use outcome labels to choose the frozen references at inference.
Custom callers can pass unlabeled `PreparedRecording` objects to `predict`.
Continuous array units are microvolts, epoch offsets are integer samples, and
recording/trial/channel identities must be stable and unique. Model checkpoints
verify channel order and epoch shape.

Constructor injection accepts `DatasetAdapter`, `PairDecomposer` and
`ModelTrainer` implementations (see `types.py`). A trainer's `fit` receives
only fitting/selection `TrialBatch` objects and returns a `FitResult`, including
validation probabilities. The pipeline independently verifies candidate BCE.
Trainer checkpoints must live under the supplied candidate/model directory.
Custom backend classes/settings must remain stable for resumption; implementers
must include additional backend settings in the config used for fingerprints.

## Verification

The focused regression run on 2026-10-21 passed 111 tests, including this
package, the analyzer/supervisor, decomposition, EEGNet, and the existing ERP
component/comparison entrypoints (three existing FastICA convergence warnings).
The earlier real-data smoke run used three channels (FP1/F3/C3), one epoch,
and five eligible subjects (sub-002/004/005/006/007). Sub-003 was excluded by raw
eligibility before splitting. All six candidates, the six-channel final model,
the raw baseline, validation/test evaluation and bundle export completed. These
settings validate execution, not predictive performance. That smoke run did
not validate full 30-channel training or CUDA/MPS execution.

On 2026-10-21 the replacement three-channel experiment completed on MPS with
37 eligible subjects, 11,815 trials and the unchanged maximum 200-epoch budget.
Its six candidates used two channels each; the final/Raw models used six/three.
All 1,671 test trials from eight subjects were reloaded without labels through
the bundle; decisions matched exactly and probabilities differed by at most
`1.2e-16`. The frozen mapping was FP1→C3, F3→FP1, C3→FP1. Raw/Selected test
AUROC was 0.8377/0.8327, AUPRC 0.4839/0.4778, F1 0.5219/0.4688, and balanced
accuracy 0.7502/0.7441. All four paired subject-bootstrap difference intervals
included zero. This run establishes end-to-end execution, without improved
classification utility or evidence of physiological denoising. The root
[README](../README.md#verified-three-channel-run-and-research-limits) records
the comparison and uncertainty; full local reports are in
`results/auto_ecmad_ern_3ch/analysis/`.

In a sandbox where default user cache directories are read-only, the verified
test command redirects library caches to temporary writable locations:

```bash
MPLCONFIGDIR=/tmp/auto_ecmad_mpl XDG_CACHE_HOME=/tmp/auto_ecmad_cache \
NUMBA_CACHE_DIR=/tmp/auto_ecmad_numba _MNE_FAKE_HOME_DIR=/tmp/auto_ecmad_mne \
conda run --no-capture-output -n eeg_pipeline python -m pytest \
  tests/test_auto_ecmad tests/test_decomposition tests/test_ml/test_cnn_model.py \
  tests/test_scripts/test_14_erp_component_eegnet.py \
  tests/test_scripts/test_16_erp_comparison_eegnet.py -q
```

Artifacts are local to the output directory: `prepared.json`, prepared caches,
trial/split manifests, exclusions, candidate checkpoints/history/diagnostics,
`candidate_scores.csv`, `selected_mapping.json`, final/raw models, frozen
thresholds, subject/trial metrics, `summary.json`, `report.md`, and a standalone
checksummed `bundle/`. The mapping is fixed at inference, while ECMAD filter
coefficients are estimated adaptively from each new recording.
