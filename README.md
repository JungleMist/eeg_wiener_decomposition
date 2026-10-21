# ECMAD — EEG Channel Matrix Adaptive Denoiser

**EEG Channel Matrix Adaptive Denoiser (ECMAD)** is an EEG denoising framework
that estimates and removes shared physical/artifact components across
anatomically motivated channel matrices while preserving locally generated EEG.
Its current core implementation uses channel-group matrix-adaptive vector Wiener
decomposition. Scripts 01–07 support both the TUH EEG Epilepsy Corpus (TUEP) and
TUH Abnormal EEG Corpus (TUAB) as downstream evaluation tracks.

---

## Table of Contents

1. [Automated ECMAD with EEGNet](#automated-ecmad-with-eegnet)
2. [Research Hypothesis](#research-hypothesis)
3. [Core Methodology](#core-methodology)
4. [Environment Setup](#environment-setup)
5. [Project Structure](#project-structure)
6. [Data Flow](#data-flow)
7. [Configuration](#configuration)
8. [Running the Pipeline](#running-the-pipeline)
9. [Physical Verification Experiments](#physical-verification-experiments)
10. [Visualization](#visualization)
11. [Testing](#testing)
12. [Package API](#package-api)

---

## ECMAD Studio (`eeg_bg_studio`)

The repository includes **ECMAD Studio**, a Chinese PySide6 desktop application
for interactive ECMAD denoising and EEG preprocessing. It supports
EDF/FIF/EEGLAB SET input, EDF/FIF output,
synchronized raw and processed waveform inspection, fixed-window or continuous
extraction, standard ICA, and ECMAD's frequency/phasegated/zerophase Wiener
modes. The Python package and executable entry point retain the compatible
internal names `eeg_bg` and `eeg_bg_studio`.

Interactive raw and processed waveforms use one shared vertical gain and fixed
channel spacing derived once from the source recording. Both panels are linked
for Y-axis navigation and apply per-channel display-only smooth baseline
centering from 20-second median anchors; this does not modify processing inputs
or exported EEG data.

EEGLAB `.set` recordings may store samples in a paired `.fdt` file. Keep the
matching `.set` and `.fdt` files in the same directory; select the `.set` file
in interactive preview, or scan their parent directory in batch processing.
When BIDS-style `*_eeg.json`, `*_channels.tsv`, `*_events.tsv`,
`*_electrodes.tsv`, and `*_coordsystem.json` sidecars are present, interactive
preview loads them automatically and provides event-type/code navigation on the
EEG timeline. For ERP-CORE ERN recordings with FCz and valid stimulus/response
events, **打开 ERN 三方法叠加** computes a response-locked Raw/ICA/ECMAD
comparison from the current filter, sampling, explicit ICA-component, and ECMAD
parameters. Standard ICA component detection remains the script 10 FP1/FP2
automatic EOG method. The modeless
window follows script 10's standard ICA and shared-trial method: `-600..400 ms` epochs,
`-400..-200 ms` baseline, common rejection decisions, incorrect-trial FCz mean,
and the incorrect-minus-correct difference wave with trial-SD bands.

```bash
conda run -n eeg_pipeline pip install -r requirements-gui.txt
conda run -n eeg_pipeline python -m eeg_bg.gui
```

The GUI and batch workflow call the same services in `eeg_bg/application/`.
Batch output mirrors the input directory tree and writes CSV/JSON manifests
containing the effective processing parameters and per-file status.

Windows x64 packaging uses PyInstaller in one-directory mode. Run the following
from PowerShell on Windows; PyInstaller must run on the target operating system.

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

The build produces `dist\eeg_bg_studio-windows-x64.zip`; users extract it and
launch `eeg_bg_studio.exe` without installing Python.

---

## Dataset Support

| Dataset | Labels | Evaluation unit | Split policy | Supported path |
|---------|--------|-----------------|--------------|----------------|
| TUEP | 0 epilepsy, 1 control | patient | stratified patient split | scripts 01–08, 21–24 |
| TUAB | 0 abnormal, 1 normal | recording | official eval = test; official train is patient-grouped into train/val | scripts 01–07 |
| ERP-CORE ERN | 0 correct response, 1 error response | trial, with subject-separated partitions and per-subject reporting | deterministic nested subject holdouts | script 25, `eeg_bg.auto_ecmad` |

For TUAB, script 01 extracts non-overlapping 20-second epochs from at most the first 20 minutes of each recording. Script 06 trains at epoch level, averages `P(normal)` within each `recording_id`, selects the record-level threshold on validation recordings, and applies that fixed threshold to official eval recordings. If one patient has both normal and abnormal recordings, they remain separate labeled evaluation units but are assigned together to train or validation.

**CNN scope:** script 08's cache-driven training workflow is **TUEP-only**. `scripts/08_train_cnn.py --config configs/tuab.yaml` exits with a clear error. The shared `cnn_model.EEGNet` backbone is also reused by the component experiments and the independent ERP-CORE automatic ECMAD backend below.

Use `configs/default.yaml` for TUEP and `configs/tuab.yaml` for TUAB. The two configs use different cache and result roots.

### Continuous TUEP component experiment (scripts 21–24)

Scripts 21–24 mirror the TUAB component experiment while retaining TUEP's
patient-level evaluation semantics. Script 21 applies 20-second, 50%-overlap
ECMAD windows independently to every contiguous seizure-buffered background
interval; intervals are never joined before filtering. Script 22 then creates
non-overlapping 20-second epochs with one raw-derived artifact mask and stores
`raw`, `specific`, `coherent`, and 38-channel `specific_coherent` (specific bank
first). Missing TUEP channels are zero-filled into the canonical 19-channel
order. Scripts 23 and 24 aggregate all recordings and epochs by patient;
`1=control` is the modeled positive class and corresponds to `01_no_epilepsy`.

```bash
conda run -n eeg_pipeline python scripts/21_cache_tuep_continuous_wiener.py --workers 4
conda run -n eeg_pipeline python scripts/22_extract_tuep_component_epochs.py --workers 4
conda run -n eeg_pipeline python scripts/23_train_tuep_component_eegnet.py --condition all --device auto
conda run -n eeg_pipeline python scripts/24_train_tuep_component_xgboost.py --condition all --feature-set base211
```

Script 23 selects its checkpoint by validation patient AUPRC and freezes a
validation patient balanced-accuracy threshold before one test evaluation.
Script 24 uses patient-grouped XGBoost training, chooses the macro-F1 threshold
on validation patients, and reports patient-level metrics and SHAP artifacts.

---

## Automated ECMAD with EEGNet

[`eeg_bg.auto_ecmad`](eeg_bg/auto_ecmad/) is an independent, extensible two-stage
training workflow. Its first adapter supports ERP-CORE response-locked ERN
classification. Script 25 only parses arguments and invokes the package; it
does not depend on scripts 01–08 or their caches. See the
[specialist guide](docs/auto_ecmad.md) for additional implementation details.

### Two-stage reference selection

For every target channel $t$, enumerate every other channel $i$. Decompose the
original preprocessed continuous target/reference pair and train a **fresh
two-channel EEGNet using only `[coherent, specific]` from the target direction**:

$$
i_t^* = \operatorname*{arg\,min}_{i \ne t}
\mathcal{L}_{\mathrm{inner\ selection}}
\left(f_{\hat\theta_{t,i}}([c_{t,i,\mathrm{coherent}},
c_{t,i,\mathrm{specific}}]), y\right).
$$

The candidate checkpoint and reference are selected by internal-selection loss.
Loss differences within `1e-6` are resolved by configured reference-channel
order. Once **all targets** finish, freeze the reference mapping and train a new
EEGNet on the interleaved components in target-channel order:

```text
target1::coherent, target1::specific, target2::coherent, target2::specific, ...
```

The final input has `2n` channels. An independent `n`-channel Raw EEGNet uses the
same subjects, trials and training protocol. Candidate weights are never reused
for the final model. Decompositions always start from the original continuous
signals; selected components are not cascaded or fused across references.
References may repeat or form reciprocal pairs. Independent target selection
does not guarantee the best joint combination or physiological denoising quality.

| Configuration | Original channels | Candidate fits | Final / Raw channels | Total fits | Default result root |
|---|---|---:|---|---:|---|
| [`erp_core_auto_ecmad_eegnet_3ch.yaml`](configs/erp_core_auto_ecmad_eegnet_3ch.yaml) | FP1, F3, C3 | 6 | 6 / 3 | 8 | `results/auto_ecmad_ern_3ch/` |
| [`erp_core_auto_ecmad_eegnet.yaml`](configs/erp_core_auto_ecmad_eegnet.yaml) | 30 EEG channels | 870 | 60 / 30 | 872 | `results/auto_ecmad_ern/` |

The three-channel overlay changes only the channel list and result directory;
it retains all eligible subjects and the full 200-epoch maximum budget. Create
another config directly under `configs/` to change `auto_ecmad.adapter.channels`
or select subjects. At least two channels and five eligible subjects are needed.

### Data isolation and default training rules

Subject splits use seed `42`: 20% test, then 20% of the remaining subjects for
external validation, then 20% of the external-training subjects for internal
selection. Holdout counts round up while retaining a nonempty fitting group.
The four subject groups are disjoint. Preparation fixes eligibility, response
events, labels, raw-derived rejection masks and unique trial IDs for every
candidate; search loads only internal-fit and internal-selection recordings.

Stage-one loss averages class-weighted BCE within each subject and then averages
equally across subjects. Weights are `w0=1`, `w1=N0/N1`, computed from the current
fitting trials only and reused for selection. Every candidate resets the model,
optimizer, RNG and data order with the same seed and budget. Stage two merges
both internal groups as external training, selects its checkpoint by external
validation AUPRC, and freezes the validation balanced-accuracy threshold. Both
final and Raw models must be frozen before test data are loaded for evaluation.

| Setting | Default |
|---|---|
| ERP preprocessing | 60 Hz notch, 0.5–45 Hz bandpass, 250 Hz sampling |
| ERN epochs / baseline | `[-0.6, 0.4] s` / `[-0.4, -0.2] s` |
| Shared artifact rejection | Raw EEG peak-to-peak threshold 250 µV; evaluated before selecting the model-channel subset |
| Continuous ECMAD | 12-second windows, 50% overlap-add, `nperseg=250` |
| ECMAD admission | `phasegated`, phase threshold **1 rad**, coherence threshold 0, power gate disabled, no protected band |
| Numerical handling | Existing stability checks and coefficient cap retained; protected windows use `coherent=0, specific=raw`, with diagnostics; nonfinite inputs/losses fail |
| Normalization | Baseline correction, then independent z-score per trial and channel |
| EEGNet | `F1=8`, `D=2`, `dropout=0.25` |
| Optimization | Adam, learning rate 0.001, weight decay 0.0001, batch size 64 |
| Stopping / scheduler | Maximum 200 epochs; patience 20; plateau patience 10, learning-rate factor 0.5 |

`specific + coherent ≈ raw` is checked before baseline/normalization. Training
consumes trial tensors `(trials, channels, times)`; reconstruction identity is
not expected after separately normalizing the two components.

### Run the three-channel workflow

Keep paired EEGLAB `.set`/`.fdt` files and ERP event annotations under
`~/Data/ERP_CORE/sub-*/eeg/`. Run from the repository root after installing the
package into `eeg_pipeline` (see [Environment Setup](#environment-setup)):

```bash
conda run --no-capture-output -n eeg_pipeline python \
  scripts/25_optimize_erp_core_ecmad_eegnet.py all \
  --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml \
  --data-dir ~/Data/ERP_CORE --device mps

conda run -n eeg_pipeline python scripts/analyze_auto_ecmad_results.py \
  --results-dir results/auto_ecmad_ern_3ch
```

Use `--device cpu` or `--device cuda` for other hardware; `auto` chooses
CUDA, MPS, then CPU. Fits run **one model at a time**. `--workers N` controls raw
recording preparation threads, not simultaneous training or GPU count. Pin CUDA
devices externally with `CUDA_VISIBLE_DEVICES` when needed.

`all` executes `prepare → search → train → evaluate`; detailed analysis is the
separate command above. Each stage can also be resumed independently, with the
same configuration and overrides on every invocation:

```bash
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py prepare --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py search --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py train --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py evaluate --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml --device mps
```

The complete CLI is `[prepare|search|train|evaluate|predict|all]`, with
`--config`, `--data-dir`, `--output-dir`, `--device`, `--workers` and `--force`.
`predict` additionally accepts `--bundle`. Omitting `--config` selects the
**30-channel, 872-fit** configuration; use the explicit three-channel config
for the smaller run.

For supervised execution that automatically analyzes successful training:

```bash
conda run --no-capture-output -n eeg_pipeline python \
  scripts/run_auto_ecmad_training_job.py \
  --config configs/erp_core_auto_ecmad_eegnet_3ch.yaml \
  --output-dir results/auto_ecmad_ern_3ch --device mps
```

This POSIX supervisor uses an advisory lock, writes `job_status.json` and
separate `training.log` / `analysis.log`, and records the exact child command,
PIDs, fixed device, timestamps and exit status. On macOS, prefix its `python`
command with `caffeinate -i` to keep an unattended run awake. Interrupting the
supervisor also stops its child and marks the job interrupted. Analysis can be
rerun independently without retraining when training is already complete.

### Artifacts, resumption and analysis

All auto_ecmad artifacts, including prepared caches, live under the configured
`paths.results_dir` or `--output-dir` (the inherited `paths.cache_dir` is not
used by this package):

| Artifact | Purpose |
|---|---|
| `resolved_config.json`, `prepared.json`, `prepared/` | Effective config, source fingerprints, pickle-free continuous caches |
| `split_manifest.csv`, `trial_manifest.csv`, `excluded_recordings.json` | Fixed participants/trials and exclusion reasons (subject groups also saved in `prepared.json`) |
| `candidates/`, `candidate_scores.csv` | Per-candidate checkpoint, completion status, history, score and decomposition diagnostics |
| `selected_mapping.json` | Frozen target/reference mapping, selection scores, top-two loss margins and channel order |
| `models/`, `summary.json`, `report.md` | Final/Raw checkpoints and histories, validation/test predictions, frozen thresholds and metrics |
| `bundle/` | Selected model, mapping, preprocessing/config, normalization and checksummed manifest |
| `analysis/` | Audited report, detailed tables, paired uncertainty and figures |

Atomic completion records and checksums protect resumption. A completed candidate
with a matching fingerprint is reused; an interrupted candidate starts again.
Fingerprints include source data, preprocessing/events, channels/trials/splits,
decomposition, `target_components_only`, training settings, package/backend code
and dependency versions. Changing these requires `all --force` or a new output
directory. Keep device overrides consistent between stages and avoid concurrent
writers to one result directory.

Detailed analysis audits the complete candidate grid, frozen mappings,
checksums, trial alignment and reported metrics before writing
`analysis/report.md` and a self-contained `analysis/report.html`. It includes:

- Trial and individual-subject AUROC, AUPRC, F1, balanced accuracy and confusion matrices.
- `model_metrics.csv`, `selected_references.csv`, candidate rankings, training lengths and top-two loss gaps.
- `paired_cluster_bootstrap.csv`: paired test-subject cluster bootstrap of Selected minus Raw, default 1,000 repetitions and seed 42.
- Six PNG/PDF figures: candidate losses, selected references, test ROC/PR, confusion matrices, training curves and calibration.

Bootstrap intervals condition on these frozen models and the observed test
subjects; they do not measure uncertainty from retraining, reference selection
or new splits. A single seed/split classification result is indirect evidence
of task utility and cannot establish removal of artifacts or preservation of
physiological sources.

### Inference and extension APIs

```bash
conda run -n eeg_pipeline python scripts/25_optimize_erp_core_ecmad_eegnet.py predict \
  --bundle results/auto_ecmad_ern_3ch/bundle \
  --data-dir /path/to/new/ERP_CORE --device cpu \
  --output-dir results/auto_ecmad_new_predictions
```

The saved reference mapping and decision threshold stay fixed. ECMAD
coefficients adapt to each new continuous recording. Outcome labels are
optional; the ERP adapter still needs stimulus/response annotations to locate
response-locked windows. The bundle validates channel order, epoch grid and
artifact checksums.

```python
from eeg_bg.auto_ecmad import AutoECMADPipeline

pipeline = AutoECMADPipeline("configs/erp_core_auto_ecmad_eegnet_3ch.yaml")
# Independent stage API: prepare(), search(), fit(), evaluate(), predict().
# run() performs preparation through evaluation.

inference = AutoECMADPipeline.from_bundle(
    "results/auto_ecmad_ern_3ch/bundle",
    data_dir="/path/to/new/ERP_CORE", device="cpu",
    output_dir="results/auto_ecmad_new_predictions",
)
predictions = inference.predict()
```

[`types.py`](eeg_bg/auto_ecmad/types.py) defines constructor-injected
`DatasetAdapter`, `PairDecomposer` and `ModelTrainer` protocols. Continuous
records use `(channels, samples)` in µV; `TrialBatch` uses
`(trials, channels, times)`, stable subject/trial IDs and optional labels.
The MNE boundary converts units. Trainer fitting receives only fit/selection
batches, with separate prediction/evaluation operations. Dataset, model and
search extensions can reuse the pipeline; the built-in backends currently
implement ERP-CORE, EEGNet and exhaustive search only. Existing ERP training
interfaces keep their previous defaults.

### Verified three-channel run and research limits

The **2026-10-21 MPS run completed all eight fits** using the full default
training budget: 37 eligible subjects, 11,815 shared trials and 8 test subjects
(1,671 trials). The frozen references were `FP1 → C3`, `F3 → FP1`, `C3 → FP1`.
Final/Raw checkpoints came from epochs 31/44 and stopped after 51/64 epochs.
Reloading the bundle without test labels reproduced every decision and matched
evaluation probabilities within `1.2e-16`.

| Test metric | Raw (3 channels) | Selected ECMAD (6 channels) | Selected − Raw, subject-bootstrap 95% CI |
|---|---:|---:|---|
| AUROC | 0.8377 | 0.8327 | −0.0050 [−0.0300, 0.0136] |
| AUPRC | 0.4839 | 0.4778 | −0.0062 [−0.0411, 0.0166] |
| F1 | 0.5219 | 0.4688 | −0.0532 [−0.1151, 0.0149] |
| Balanced accuracy | 0.7502 | 0.7441 | −0.0061 [−0.0519, 0.0276] |

This run demonstrates a working search/training/evaluation/inference path;
these metrics do **not show improved classification utility**. More false
positives accompanied higher error-response recall. Validation checkpoint and
threshold selection, only eight test subjects, and a single seed/split limit
the conclusions. Independent target selection also leaves joint optimality
unresolved. Review the local `results/auto_ecmad_ern_3ch/analysis/report.html`
for the complete analysis; generated results/models are gitignored.

The 30-channel run in `results/auto_ecmad_ern/` was stopped at the user's request
after 43 completed candidates and has no complete final result. The earlier
`results/auto_ecmad_real_smoke/` used five subjects and one epoch; it verifies
execution only and is distinct from the completed three-channel experiment.

Regression tests cover directed decomposition, phase/numerical safeguards,
reconstruction, subject-equal loss, strict two-channel candidates, split/trial
isolation, fresh training, deterministic ranking, cache corruption/invalidation,
unlabeled bundle reload, alternative adapter/trainer implementations and
EEGNet forward/backward passes at 2/60 channels:

```bash
conda run -n eeg_pipeline python -m pytest tests/test_auto_ecmad -q
```

---

## Research Hypothesis

ECMAD models inter-channel coherent components in background EEG as
**non-neural physical interference** — volume conduction, reference electrode
contamination, and shared environmental noise. Removing these through its
frequency-dependent vector Wiener implementation reveals the **specific**
(locally generated, cortical) signal component.

**Decomposition identity (exact):**

```
x_i(t) = specific_i(t) + coherent_i(t)

coherent_i(t) = IFFT( Σ_j h_ij(f) · X_j(f) )
specific_i(t) = x_i(t) - coherent_i(t)
```

where `h_ij(f)` is the optimal Wiener filter coefficient estimated from the cross-power spectral density matrix of each bilateral electrode pair.

---

## Core Methodology

### Signal Model

The observed EEG at electrode $i$ within a muscle-artifact conduction group is modelled as a linear mixture of a locally generated signal and attenuated copies from neighbouring electrodes:

$$x_i(t) = s_i(t) + \sum_{j \neq i} h_{ij}(t) * x_j(t)$$

where $s_i(t)$ is the **specific** (cortical) component and $h_{ij}(t)$ is the impulse response of the physical conduction path from electrode $j$ to electrode $i$. The goal is to recover $s_i(t)$ for every electrode in each group.

---

### Channel Groups

The filter is applied independently to six anatomically motivated **conduction groups**, each corresponding to a known muscle-artifact pathway. Channels not in any group (`passthrough`) are left unchanged.

| Group | Channels | Pathway |
|-------|----------|---------|
| G1 | FP1, FP2 | Symmetric frontalis (facial) |
| G2 | F7, T3 | Left sternocleidomastoid (SCM) |
| G3 | T3, T5, O1 | Left posterior neck (3-channel chain) |
| G4 | O1, O2 | Bilateral occipitalis |
| G5 | F8, T4 | Right SCM |
| G6 | T4, T6, O2 | Right posterior neck (3-channel chain) |
| — | F3, F4, C3, C4, P3, P4, Fz, Cz, Pz | Passthrough (never filtered) |

3-channel chains (G3, G6) model the fact that the middle electrode (T5 or T6) receives artifact from both its neighbours; each channel in the group is processed using all remaining channels in that group as references.

---

### Step 1 — Cross-PSD Estimation

For each group of $K$ channels, the full $K \times K$ **cross-power spectral density matrix** is estimated using Welch's method with a boxcar (rectangular) window:

$$S_{ij}(f) = \frac{1}{L} \sum_{l=1}^{L} X_i^{(l)}(f) \cdot \overline{X_j^{(l)}(f)}$$

where $L$ is the number of non-overlapping segments and $X_i^{(l)}(f)$ is the DFT of segment $l$ of channel $i$.

**Implementation parameters** (default config):

| Parameter | Value | Derivation |
|-----------|-------|-----------|
| `nperseg` | 250 samples | $f_s \times 2\,\text{s} = 125 \times 2$ |
| Segments per epoch | 4 | $1000\,\text{samples} / 250$ |
| Frequency resolution | 0.5 Hz | $f_s / \text{nperseg} = 125 / 250$ |
| Window | boxcar | Matches the rfft applied during filter application |

The boxcar window is essential: using any other window would introduce a mismatch between the estimated filter grid and the full-epoch rfft used in Step 3, violating the exact decomposition identity.

**Code:** `eeg_bg/decomposition/wiener.py` → `estimate_cross_psd()`

---

### Step 2 — Coherence Gate

Before estimating a filter, the maximum pairwise coherence across all channel pairs in the group is computed over the target frequency band:

$$\gamma^2_{ij}(f) = \frac{|S_{ij}(f)|^2}{S_{ii}(f) \cdot S_{jj}(f)}, \qquad \gamma^2_{ij} \in [0, 1]$$

$$C_\text{max} = \max_{\substack{i,j \in \text{group} \\ i \neq j}} \max_{f \in [f_\text{lo},\, f_\text{hi}]} \gamma^2_{ij}(f)$$

If $C_\text{max} < \theta_\text{coh}$ (default 0.15), the group is **skipped** — no filtering is applied and those channels pass through unchanged. This avoids fitting noise in groups where no shared artifact source is detectable.

---

### Step 3 — Wiener Filter Estimation

For each channel $i$ in the group (treated as the target), the remaining $K-1$ channels form the reference set. The optimal **minimum mean-squared-error (MMSE) Wiener filter** solves the Wiener–Hopf equation at each frequency bin $f$:

$$\mathbf{h}_i(f) = \mathbf{S}_\text{ref}(f)^{-1}\, \mathbf{s}_{i,\text{ref}}(f)$$

where:
- $\mathbf{S}_\text{ref}(f) \in \mathbb{C}^{(K-1)\times(K-1)}$ — cross-PSD sub-matrix of the reference channels
- $\mathbf{s}_{i,\text{ref}}(f) \in \mathbb{C}^{K-1}$ — cross-PSD vector between target $i$ and the references
- $\mathbf{h}_i(f) \in \mathbb{C}^{K-1}$ — filter coefficients (one per reference channel, per frequency bin)

**Tikhonov regularisation** is applied to stabilise the solve for near-singular matrices (common in 3-channel chains where reference channels are highly correlated):

$$\mathbf{S}_\text{ref,reg}(f) = \mathbf{S}_\text{ref}(f) + \varepsilon(f)\,\mathbf{I}$$

$$\varepsilon(f) = \lambda \cdot \max\!\left(\overline{\operatorname{diag}\bigl(\operatorname{Re}[\mathbf{S}_\text{ref}(f)]\bigr)},\; 10^{-30}\right), \qquad \lambda = 10^{-4}$$

The regularisation is relative (proportional to the mean diagonal power), so it scales correctly across subjects with very different signal amplitudes.

**Stability gate:** After solving, if $\max_{j,f} |h_{ij}(f)| > 50$ for any channel in the group, the entire group is treated as unstable and skipped — even after regularisation the matrix was effectively singular.

**Code:** `eeg_bg/decomposition/wiener.py` → `compute_wiener_filter()`

---

### Step 4 — Filter Application

The coherent component is computed in the frequency domain by summing the filtered reference channels:

$$\hat{X}_i^\text{coherent}(f) = \sum_{j \neq i} h_{ij}(f) \cdot X_j(f)$$

$$x_i^\text{coherent}(t) = \operatorname{IFFT}\!\left[\hat{X}_i^\text{coherent}(f)\right]$$

$$x_i^\text{specific}(t) = x_i(t) - x_i^\text{coherent}(t)$$

When `nperseg < n_times` (the Welch window is shorter than the epoch), the filter coefficients are **linearly interpolated** from the Welch frequency grid (size $\lfloor\text{nperseg}/2\rfloor + 1$) to the full rfft grid (size $\lfloor n_\text{times}/2\rfloor + 1$) before application. The interpolation is on real and imaginary parts separately.

Because the coherent component is defined as $x_i - x_i^\text{specific}$ by construction, the **decomposition identity holds exactly regardless of interpolation accuracy**:

$$x_i^\text{specific}(t) + x_i^\text{coherent}(t) \equiv x_i(t)$$

**Code:** `eeg_bg/decomposition/wiener.py` → `apply_wiener_filter()`

---

### Scalar Ablation Baseline

The `--mode scalar` variant collapses the frequency-dependent filter to a single **real scalar** per reference channel by averaging $h_{ij}(f)$ over the target band:

$$\bar{h}_{ij} = \frac{1}{|F|} \sum_{f \in F} \operatorname{Re}[h_{ij}(f)], \qquad F = \{f : f_\text{lo} \le f \le f_\text{hi}\}$$

The coherent signal is then computed entirely in the time domain:

$$x_i^\text{coherent}(t) = \sum_{j \neq i} \bar{h}_{ij} \cdot x_j(t)$$

This is otherwise identical to the frequency mode (same coherence gate, same PSD estimation, same regularisation). It serves as the ablation baseline for V3: if $|h_{ij}(f)|$ varies significantly across frequency (V3 passes), the frequency-dependent model should outperform the scalar one on the downstream classification task.

**Code:** `eeg_bg/decomposition/wiener_scalar.py`

---

### Summary of Filter Parameters

| Symbol | Config key | Default | Role |
|--------|-----------|---------|------|
| $f_s$ | `preprocessing.target_sfreq` | 125 Hz | Sampling rate after resampling |
| $\text{nperseg}$ | `wiener.nperseg` | 250 | Welch window length (samples) |
| $[f_\text{lo}, f_\text{hi}]$ | `wiener.freq_band` | [0.5, 40.0] Hz | Filter estimation and coherence gate band |
| $\theta_\text{coh}$ | `wiener.coherence_threshold` | 0.15 | Coherence gate threshold |
| $M_\text{max}$ | `wiener.filter_magnitude_threshold` | 50.0 | Stability gate: max allowed $\|h_{ij}(f)\|$ |
| $\lambda$ | — (hardcoded) | $10^{-4}$ | Tikhonov regularisation factor |

---

## Environment Setup

**Prerequisite:** Anaconda installed at `C:\ProgramData\anaconda3\`

```bash
# Activate the project environment
conda activate eeg_pipeline

# Install the package in editable mode (run once)
pip install -e .
```

**Core dependencies** (pinned in `requirements.txt`):

| Package | Version | Role |
|---------|---------|------|
| numpy | 2.4.6 | Array operations |
| scipy | 1.17.1 | Welch PSD, CSD, coherence |
| mne | 1.11.0 | EDF I/O, ICA |
| scikit-learn | 1.8.0 | GridSearchCV, StandardScaler, metrics |
| joblib | 1.5.3 | Model and scaler serialisation |
| pandas | 2.3.3 | Result DataFrames, CSV output |
| matplotlib | 3.10.8 | All visualization |
| pyyaml | 6.0.3 | Configuration loading |
| pytest | 9.0.2 | Test suite |
| tqdm | 4.67.3 | Pipeline progress bars |
| xgboost | 3.2.0 | XGBoost classifier, GridSearchCV, early stopping |
| shap | 0.51.0 | TreeExplainer SHAP values, beeswarm summary plots |
| pywavelets | ≥1.4 | DWT features (`eeg_bg/features/wavelet.py` — not called from `extraction.py`, see Step 6) |
| torch | CPU or CUDA build | EEGNet CNN pipeline (script 08) |

---

## Project Structure

```
D:\eeg_background_seizure\
│
├── eeg_bg/                          ← Core Python package
│   ├── config/
│   │   └── settings.py              ← YAML loader, path resolution
│   ├── io/
│   │   ├── dataset.py               ← Subject index builder, train/val/test splits
│   │   ├── edf_reader.py            ← EDF loading, channel selection, resampling
│   │   ├── annotation.py            ← .csv_bi parser, background interval extraction
│   │   └── cache.py                 ← .npz read/write, SHA-256 cache keys
│   ├── preprocessing/
│   │   ├── epoch.py                 ← 8-second epoch slicing, artifact rejection, bandpass
│   │   └── reference.py             ← Montage reference detection (AR vs LE)
│   ├── decomposition/
│   │   ├── wiener.py                ← Vector Wiener filter (core method)
│   │   ├── wiener_scalar.py         ← Fixed scalar ablation (comparison baseline)
│   │   └── ica.py                   ← FastICA decomposition via MNE (comparison method)
│   ├── auto_ecmad/                  ← Two-stage EEGNet-guided directed-reference search
│   │   ├── types.py, pipeline.py    ← Dataset/model protocols, isolation and stage scheduling
│   │   ├── decomposition.py        ← Directed continuous ECMAD and ordered trial components
│   │   ├── search.py, training.py   ← Candidate ranking, independent EEGNet fitting/evaluation
│   │   ├── artifacts.py            ← Fingerprints, atomic completion and inference bundles
│   │   └── adapters/erp_core.py     ← ERN preparation, shared events and raw trial masks
│   ├── verification/
│   │   ├── coherence.py             ← V1: pairwise coherence reduction
│   │   └── transitivity.py          ← V2: transitivity constraint, V3: frequency variation
│   ├── visualization/
│   │   ├── filter_plots.py          ← |h(f)| and ∠h(f) curves
│   │   ├── coherence_plots.py       ← Coherence heatmaps, reduction boxplots, signal traces
│   │   ├── verification_plots.py    ← V1/V2/V3 summary figures, ICA vs Wiener comparison
│   │   └── waveform_plots.py        ← Stacked multi-channel waveform comparison
│   ├── features/
│   │   ├── band_power.py            ← Relative band power (5 EEG bands, boxcar Welch)
│   │   ├── hjorth.py                ← Hjorth activity / mobility / complexity
│   │   ├── spectral_entropy.py      ← Shannon entropy of normalized PSD
│   │   ├── asymmetry.py             ← Hemispheric asymmetry (8 pairs × 5 bands)
│   │   ├── extraction.py           ← extract_epoch_features(), build_dataset(), FEATURE_NAMES (211-dim)
│   │   └── wavelet.py, connectivity.py, complexity.py, temporal_stats.py
│   │                                  ← Disconnected feature blocks (not called from extraction.py;
│   │                                     kept for potential future use, see History note above)
│   └── ml/
│       ├── xgb_pipeline.py          ← train_xgboost(), subject_level_predict(), evaluate_subject_level(), find_optimal_threshold()
│       ├── shap_analysis.py         ← compute_shap_values(), aggregate/plot functions
│       ├── cnn_model.py             ← EEGNet — compact EEG classification CNN (Lawhern et al. 2018)
│       ├── cnn_dataset.py           ← EEGEpochDataset — reads the same cache/ trees as build_dataset(), yields raw tensors
│       └── cnn_pipeline.py          ← cnn_predict_epochs(), train_cnn() — parallel CNN path, no hand-crafted features
│
├── scripts/
│   ├── 01_extract_epochs.py         ← Step 1: EDF → cached epochs (.npz)
│   ├── 02_run_wiener.py             ← Step 2: cached epochs → Wiener decomposition
│   ├── 03_run_ica.py                ← Step 3: cached epochs → ICA decomposition
│   ├── 04_run_verification.py       ← Step 4: V1/V2/V3 verification → CSV reports
│   ├── 05_run_visualization.py      ← Step 5: cached results → waveform + PSD figures
│   ├── 06_train_xgboost.py         ← Step 6: feature extraction + XGBoost × 3 conditions + SHAP
│   ├── 07_organize_experiment.py   ← Step 7: archive config + results → experiments/<timestamp>/
│   ├── 08_train_cnn.py             ← Step 8: EEGNet CNN trained directly on raw epoch tensors → results/cnn/
│   ├── 25_optimize_erp_core_ecmad_eegnet.py ← Automatic ECMAD prepare/search/train/evaluate/predict
│   ├── run_auto_ecmad_training_job.py     ← Supervise training, then run analysis
│   └── analyze_auto_ecmad_results.py      ← Audit artifacts, report metrics/uncertainty and figures
│
├── configs/
│   ├── default.yaml                 ← Shared pipeline parameters (including ml: section)
│   ├── erp_core_auto_ecmad_eegnet.yaml     ← 30-channel exhaustive automatic ECMAD configuration
│   └── erp_core_auto_ecmad_eegnet_3ch.yaml ← FP1/F3/C3 overlay with unchanged training budget
│
├── tests/
│   ├── conftest.py                  ← Synthetic EEG fixtures (no real data needed)
│   ├── test_config.py
│   ├── test_io/                     ← dataset, edf_reader, annotation, cache
│   ├── test_preprocessing/          ← epoch slicing, reference detection
│   ├── test_decomposition/          ← wiener, wiener_scalar, ica
│   ├── test_auto_ecmad/             ← Two-stage isolation, training, resume, inference and analysis
│   ├── test_verification/           ← coherence V1, transitivity V2/V3
│   ├── test_features/               ← band_power, hjorth, spectral_entropy, extraction, asymmetry, wavelet, connectivity, complexity, temporal_stats (72 tests)
│   ├── test_ml/                     ← xgb_pipeline, shap_analysis, CNN model/dataset/training (31 tests)
│   └── test_scripts/                ← script 06 shape guard (1 test)
│
├── experiments/                     ← Auto-generated, git-ignored
│   └── YYYY-MM-DD_HHMMSS[_<name>]/ ← One folder per archived run (script 07)
│       ├── config.yaml              ← Copy of config used
│       ├── experiment.json          ← Config snapshot + all metrics (machine-readable)
│       ├── report.md                ← Human-readable summary with results table
│       ├── comparison_summary.csv   ← Re-derived from present conditions
│       ├── shap_comparison.png      ← Re-generated from per-condition SHAP data
│       └── {raw,ica,wiener}/        ← Per-condition metrics JSONs + SHAP plots
│
├── cache/                           ← Auto-generated, git-ignored
│   ├── epochs/                      ← Per-subject .npz epoch files + index.csv
│   ├── wiener_frequency/            ← Wiener-decomposed epochs
│   ├── wiener_scalar/               ← Scalar-ablation epochs
│   ├── ica/                         ← ICA-cleaned epochs
│   └── features/                    ← Extracted feature matrices {condition}_{split}.npz
│
├── docs/
│   └── Developer_qa.md              ← Developer Q&A (epoch validity, .npz schema, …)
│
├── results/                         ← Figure and CSV outputs, git-ignored
│   ├── figures/{subject_id}/        ← PNG outputs from script 05
│   │   ├── waveform_comparison.png
│   │   └── psd_comparison.png
│   ├── verification/                ← CSV reports from script 04
│   │   ├── v1_coherence.csv
│   │   ├── v2_transitivity.csv
│   │   └── v3_frequency_variation.csv
│   ├── xgboost/                     ← Outputs from script 06
│   │   ├── {raw,ica,wiener}/        ← model.joblib, scaler.joblib, metrics JSONs,
│   │   │                               predictions CSVs, SHAP values + plots
│   │   ├── comparison_summary.csv
│   │   └── shap_comparison.png
│   └── cnn/                         ← Outputs from script 08
│       ├── {raw,ica,wiener}/        ← best_model.pt, best_params.json, metrics JSONs,
│       │                               predictions CSVs
│       └── comparison_summary.csv
├── setup.py
├── requirements.txt
└── pytest.ini
```

---

## Data Flow

```
D:\EEGdata\TUEP\v3.1.0\
  00_epilepsy/{subject}/{session}/01_tcp_ar/*.edf   (+ .csv_bi)
  01_no_epilepsy/{subject}/{session}/01_tcp_ar/*.edf (+ .csv_bi)
          │
          ▼  Script 01
    [io] build_subject_index  →  filter AR montages  →  assign train/val/test
    [io] load_edf             →  bandpass filter  →  resample to 125 Hz
    [io] extract_bckg_intervals  →  full recording minus seizure ±30 s buffers
    [preprocessing] slice_epochs  →  reject artifacts > 200 µV
          │
          ▼  cache/epochs/{subject_id}/{cache_key}.npz
          │     arrays: epochs (n, 19, 1000)  ch_names  label  subject_id  split
          │
     ┌────┴──────────────────────────────┬─────────────────────────────────────┐
     ▼                                   ▼                                     ▼
  Script 02                          Script 03                             Script 04
  Wiener                                ICA                        [verification] V1/V2/V3
  decompose                         fit + apply                    (re-runs decompose_epoch
     │                                   │                         inline; reads only epochs)
     ▼                                   ▼                                     │
cache/wiener_frequency/   cache/wiener_scalar/   cache/ica/                    ▼
  arrays: specific  coherent  label  subject_id  split               results/verification/
          │                                          │               v1_coherence.csv
          └────────────────────┬─────────────────────┘               v2_transitivity.csv
                               │                                     v3_frequency_variation.csv
               ┌───────────────┴───────────────┐
               ▼                               ▼
           Script 05                       Script 06
    [visualization] load epoch       [features] extract_epoch_features
      + wiener_frequency + ica          × 3 conditions (211 features/epoch)
    plot_multichannel_comparison       [ml] train_xgboost (GridSearchCV
    plot_psd_comparison                     + early stopping on val)
               │                       subject_level_predict
               ▼                       evaluate_subject_level (AUROC/F1/Acc)
    results/figures/                   compute_shap_values
      {subject_id}/                             │
        waveform_comparison.png                 ▼
        psd_comparison.png             results/xgboost/
                                         {condition}/model.joblib
                                         test_metrics.json
                                         shap_summary.png
                                         comparison_summary.csv
                                         shap_comparison.png
```

**Script 08 is TUEP-only.** For TUEP it runs independently in parallel with script 06, reads the same `cache/{epochs,wiener_frequency,ica}/` trees via `EEGEpochDataset`, and trains an EEGNet CNN per condition. TUAB users should stop at script 07.

### Cache File Schema

Every `.npz` at `cache/epochs/{subject_id}/{cache_key}.npz` contains:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `epochs` | `(n_epochs, 19, 1000)` | float64 | Signal in µV |
| `ch_names` | `(19,)` | str | Channel names |
| `label` | scalar | int | 0 = epilepsy, 1 = control |
| `subject_id` | scalar | str | Anonymized patient ID |
| `split` | scalar | str | `train` / `val` / `test` |

New caches also carry `dataset_name`, `patient_id`, `recording_id`, `evaluation_id`, `class_name`, `source_partition`, and `n_epochs`. `subject_id` remains a compatibility alias of `evaluation_id`: a label-prefixed patient ID for TUEP and the EDF recording stem for TUAB.

Wiener/ICA output `.npz` files have the same structure but replace `epochs` with `specific` (and `coherent` for Wiener).

---

## Configuration

All parameters live in `configs/default.yaml`. Edit this file to change dataset paths, preprocessing settings, or method hyperparameters.

```yaml
paths:
  data_root: "D:/EEGdata/TUEP/v3.1.0"   # ← Set to your TUEP root
  cache_dir:   "cache"                    # relative to project root
  results_dir: "results"

dataset:
  active: "tuep"                  # tuep | tuab
  tuep:
    reference_scheme: "ar"
    montage_dir: "01_tcp_ar"
    classes:
      epilepsy: {folder: "00_epilepsy", label: 0}
      control: {folder: "01_no_epilepsy", label: 1}
  tuab:
    edf_dir: "edf"
    reference_scheme: "ar"
    montage_dir: "01_tcp_ar"
    train_partition: "train"
    eval_partition: "eval"
    validation_fraction: 0.10
    max_recording_sec: 1200.0
    classes:
      abnormal: {folder: "abnormal", label: 0}
      normal: {folder: "normal", label: 1}

split:
  train: 0.70
  val:   0.10
  test:  0.20
  random_seed: 42                  # subject-level split, reproducible

preprocessing:
  target_sfreq:         125        # Hz — resample target
  bandpass:             [0.5, 40.0]
  epoch_length_sec:     8.0        # → 1000 samples per epoch at 125 Hz
  artifact_threshold_uv: 200.0     # epochs exceeding this are discarded
  seizure_buffer_sec:   30.0       # exclusion zone around annotated seizures

channels:
  channel_groups:                  # G1–G6 movement-artifact conduction paths
    - [FP1, FP2]                   # G1 – symmetric facial (frontalis)
    - [F7,  T3]                    # G2 – left SCM
    - [T3,  T5, O1]                # G3 – left posterior neck (3-ch chain)
    - [O1,  O2]                    # G4 – bilateral occipitalis
    - [F8,  T4]                    # G5 – right SCM
    - [T4,  T6, O2]                # G6 – right posterior neck (3-ch chain)
  passthrough: [F3, F4, C3, C4, P3, P4, Fz, Cz, Pz]  # never filtered

wiener:
  nperseg:             250         # Welch segment length → 4 segments/epoch, 0.5 Hz resolution
  freq_resolution_hz:  0.5         # used by V1 coherence estimation
  coherence_threshold: 0.15        # skip group if max pairwise coherence < this
  freq_band:           [0.5, 40.0] # coherence gate + filter estimation band

ica:
  n_components:          19
  artifact_corr_threshold: 0.8     # correlation with FP1/FP2 proxy → artifact
  random_state:          42

verification:
  v2_transitivity_amp_threshold:   0.1    # ε_amp pass criterion
  v2_transitivity_phase_threshold: 0.392  # π/8 rad phase pass criterion
  v3_freq_variation_threshold:     0.20   # (max-min)/mean |h(f)| > 20% → frequency-dependent
```

### Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| `nperseg = 250` | = epoch / 4 (0.5 Hz resolution) | Four-segment Welch → variance reduction in cross-PSD; filter interpolated to full rfft grid; `specific + coherent = raw` guaranteed algebraically |
| Coherence gate `0.15` | Skip groups with low coherence | Avoids fitting noise; groups without a true shared source are skipped |
| Rectangular window | `window='boxcar'` | Matches the FFT applied during filter application; no spectral leakage mismatch |
| Channel groups G1–G6 | 6 groups (2- and 3-channel), not all pairs | Physically motivated: each group models a specific movement-artifact conduction pathway |

---

## Running the Pipeline

> **Always activate the environment first:**
> ```bash
> conda activate eeg_pipeline
> ```

### Step 1 — Extract Background Epochs

Traverses the TUEP dataset, filters `.csv_bi` annotations for background segments, rejects artifacts, and caches clean 8-second epochs to disk.

```bash
python scripts/01_extract_epochs.py
# or with custom config:
python scripts/01_extract_epochs.py --config configs/default.yaml
# force recompute even if cache exists:
python scripts/01_extract_epochs.py --force
```

**Output:** `cache/epochs/index.csv` (subject index with splits) and one `.npz` per EDF file under `cache/epochs/{subject_id}/`.

---

### Step 2 — Run Wiener Decomposition

Loads cached epochs and applies the vector Wiener filter to each bilateral pair.

```bash
# Frequency-dependent Wiener (default, core method):
python scripts/02_run_wiener.py --mode frequency

# Fixed scalar ablation (comparison baseline):
python scripts/02_run_wiener.py --mode scalar

# Force recompute:
python scripts/02_run_wiener.py --mode frequency --force
```

**Output:** `cache/wiener_frequency/` or `cache/wiener_scalar/` mirroring the epoch cache layout.

**What `--mode frequency` does:** For each bilateral pair, estimates `S(f)` (cross-PSD matrix) via Welch, solves `h(f) = S_ref(f)^{-1} · s_cross(f)` at each frequency bin, applies in the frequency domain. Guarantees `specific + coherent = raw` exactly.

**What `--mode scalar` does:** Same filter estimation, but collapses `h(f)` to a single complex scalar per reference channel (mean over frequency band), then applies in the time domain. Ablation baseline to test whether frequency-dependence matters.

---

### Step 3 — Run ICA Decomposition

Fits FastICA on all epochs for each subject, automatically identifies artifact components by correlation with the FP1/FP2 frontal proxy, and removes them.

```bash
python scripts/03_run_ica.py
python scripts/03_run_ica.py --force   # recompute
```

**Output:** `cache/ica/` with cleaned epochs.

**Artifact detection logic:** Each ICA component's time course is correlated with `mean(FP1, FP2)`. Components with `|r| > artifact_corr_threshold` (default 0.8) are excluded.

---

### Step 4 — Run Physical Verification

Runs V1/V2/V3 experiments and saves CSV result tables.

```bash
python scripts/04_run_verification.py
```

**Output** in `results/`:

| File | Columns | Purpose |
|------|---------|---------|
| `v1_coherence.csv` | `subject_id, epoch_idx, ch_i, ch_j, coh_pre, coh_post, reduction` | Coherence purity before/after decomposition |
| `v2_transitivity.csv` | `subject_id, epoch_idx, triplet, eps_amp, eps_phase` | Single-source transitivity constraint errors |
| `v3_frequency_variation.csv` | `subject_id, epoch_idx, pair, channel, freq_variation, amp_mean, amp_std` | Frequency-dependence of `\|h(f)\|` |

---

### Step 5 — Generate Waveform Figures

Loads one representative epoch per subject from the epoch, Wiener, and ICA caches
and saves a stacked 19-channel waveform comparison (Raw | Wiener specific | ICA cleaned).
No decomposition is re-run — arrays are read directly from cached `.npz` files.

```bash
# All subjects, epoch 0 (default):
python scripts/05_run_visualization.py

# Limit to first 5 subjects, choose epoch 2:
python scripts/05_run_visualization.py --n-subjects 5 --epoch-idx 2

# Optionally export capped EDF files for side-by-side review:
python scripts/05_run_visualization.py --export-edf --export-edf-max-epochs 3
```

**Output:** `results/figures/{subject_id}/waveform_comparison.png` and
`results/figures/{subject_id}/psd_comparison.png`. With `--export-edf`, up to
`export_edf_max_epochs` (`>=1`) epochs per subject are also written under
`results/figures/{subject_id}/edf/epoch_{i}/{condition}.edf`.

Each figure contains up to three panels (panels are omitted if the corresponding cache
does not yet exist):

| Panel | Source | Content |
|-------|--------|---------|
| Raw | `cache/epochs/` | Unprocessed bandpass-filtered EEG |
| Wiener specific | `cache/wiener_frequency/` | Coherent component removed |
| ICA cleaned | `cache/ica/` | Artifact components projected out |

---

---

### Step 6 — Train XGBoost Classifiers + SHAP Analysis

Extracts 211 handcrafted features per epoch — 171 per-channel features (relative
band power × 5 bands, Hjorth parameters × 3, spectral entropy, for each of 19
channels) plus 40 hemispheric asymmetry features (8 symmetric pairs × 5 bands) —
under the configured preprocessing conditions, trains an XGBoost classifier per condition
via 5-fold GridSearchCV + early-stopping refit on the validation set, evaluates
at patient level for TUEP or recording level for TUAB, and generates SHAP
feature importance comparison figures.

*History:* this 211-dim vector is the original feature set. It was later expanded
to 1070 dims (adding wavelet DWT, connectivity, complexity, and multi-scale
temporal-stats blocks — see `eeg_bg/features/{wavelet,connectivity,complexity,
temporal_stats}.py`), then reverted back to the original 211-dim vector to curb
overfitting risk against the ~124-subject training set. Those four modules and
their unit tests remain in the codebase, disconnected from `extraction.py`, for
potential future use.

```bash
# All three conditions (raw / ica / wiener) — default
python scripts/06_train_xgboost.py

# Single condition
python scripts/06_train_xgboost.py --condition wiener

# Force re-extraction of features (ignore feature cache)
python scripts/06_train_xgboost.py --force
```

**Output:** `results/xgboost/{condition}/` — model, metrics, per-subject
predictions, SHAP values and plots; `results/xgboost/shap_comparison.png` —
2 × 3 publication figure comparing SHAP band/channel importance across the three
conditions (Raw C | ICA B | Wiener A).

---

### Step 7 — Organize Experiment Archive

Captures a consistent snapshot of whatever condition results currently exist in
`results/xgboost/` and writes them into a timestamped folder under `experiments/`.
Unlike the top-level `comparison_summary.csv` and `shap_comparison.png` produced by
script 06 (which reflect only the last `--condition` run), script 07 **re-derives**
the summary CSV and **re-generates** the SHAP comparison figure from whichever
conditions are actually present at archive time — so a partial re-run never
produces a stale summary.

```bash
# Archive with an optional human label
python scripts/07_organize_experiment.py --name wiener-test

# Archive using a non-default config
python scripts/07_organize_experiment.py --config configs/local.yaml --name ablation
```

**Output:** `experiments/YYYY-MM-DD_HHMMSS[_<name>]/`

| File | Description |
|------|-------------|
| `config.yaml` | Copy of the config used for reproducibility |
| `experiment.json` | Config snapshot + all found metrics (machine-readable) |
| `report.md` | Human-readable table of key parameters and AUROC / F1 / Acc results |
| `comparison_summary.csv` | Re-derived from the conditions that are present |
| `shap_comparison.png` | Re-generated via `plot_shap_comparison()` |
| `{raw,ica,wiener}/` | Per-condition metrics JSONs + SHAP summary plots |

Safe to run after a partial condition re-run (e.g. `--condition wiener` in step 6)
— the archive will contain only the conditions that have complete results.

Script 07 also archives `results/cnn/` (script 08's output) alongside `results/xgboost/` whenever CNN results are present, adding a `cnn/{raw,ica,wiener}/` folder and a CNN results table in `report.md`.

---

### Step 8 — Train EEGNet CNN

A parallel classification path that trains directly on raw `(19, 1000)` epoch
tensors — no hand-crafted features. Reads the same `cache/{epochs,
wiener_frequency,ica}/` trees as script 06 and trains an independent EEGNet
(Lawhern et al. 2018) per condition: weighted BCE loss for class imbalance,
Adam optimizer, `ReduceLROnPlateau`, and early stopping on validation AUROC.
Has no dependency on script 06 and can run any time after 01 (+02/03 for the
wiener/ica conditions).

```bash
# All three conditions (raw / ica / wiener) — default
python scripts/08_train_cnn.py

# Single condition
python scripts/08_train_cnn.py --condition wiener

# Force re-training even if output already exists
python scripts/08_train_cnn.py --force
```

**Output:** `results/cnn/{condition}/` — `best_model.pt` (EEGNet state dict),
`best_params.json`, val/test metrics JSONs, val/test predictions CSVs;
`results/cnn/comparison_summary.csv` once all three conditions have been
trained.

---

### Pipeline Dependency Graph

Script 01 is the only strict prerequisite. Everything downstream of it has no mutual dependency and can run in parallel.

```
EDF files
    │
    ▼
  01_extract_epochs              →  cache/epochs/
    │
    ├──────────────────────────────────┬──────────────────────────────────┐
    ▼                                  ▼                                  ▼
  02_run_wiener                     03_run_ica                  04_run_verification
  cache/wiener_frequency|scalar/    cache/ica/                  results/verification/
    │                                  │                        [independent of 02/03;
    └──────────────┬───────────────────┘                        re-runs decompose_epoch
                   │                                             inline from cache/epochs/]
          ┌────────┴────────┐
          ▼                 ▼
  05_run_visualization   06_train_xgboost
  results/figures/       results/xgboost/
                                 │
                                 ▼
                         07_organize_experiment    (run after any subset of 02/03/06)
                         experiments/<timestamp>/
```

**Script 06 and script 08 per-condition dependencies** (identical for both — script 08 is not shown in the diagram above to keep it readable, but it hangs off `cache/epochs/` in parallel with 05/06/07, with no dependency on script 06):

| `--condition` | Requires |
|---------------|----------|
| `raw` | 01 only |
| `wiener` | 01 + 02 |
| `ica` | 01 + 03 |
| `all` (default) | 01 + 02 + 03 |

**Script 07** can be run after any subset of conditions from script 06 and/or script 08 have completed. It discovers whichever of `{raw, ica, wiener}` have results in `results/xgboost/` and `results/cnn/` and archives only those.

**Script 05** reads `cache/wiener_frequency/` and `cache/ica/` optionally — panels are omitted if those caches are absent, so it can be run after 01 alone for raw-only figures.

---

### Complete Pipeline (sequential)

```bash
conda activate eeg_pipeline
python scripts/01_extract_epochs.py
python scripts/02_run_wiener.py --mode frequency
python scripts/02_run_wiener.py --mode scalar
python scripts/03_run_ica.py
python scripts/04_run_verification.py
python scripts/05_run_visualization.py
python scripts/06_train_xgboost.py
python scripts/08_train_cnn.py
python scripts/07_organize_experiment.py --name full-run
```

### Complete Pipeline (parallel)

```bash
conda activate eeg_pipeline
python scripts/01_extract_epochs.py

# Steps 2–4: all read only from cache/epochs/ — run concurrently
python scripts/02_run_wiener.py --mode frequency &
python scripts/02_run_wiener.py --mode scalar    &
python scripts/03_run_ica.py                     &
python scripts/04_run_verification.py            &
wait

# Steps 5, 6, 8: no dependency on each other — run concurrently
python scripts/05_run_visualization.py &
python scripts/06_train_xgboost.py     &
python scripts/08_train_cnn.py         &
wait

# Step 7: archive results after steps 6 and 8 complete
python scripts/07_organize_experiment.py --name full-run
```

---

## Physical Verification Experiments

### V1 — Decomposition Purity

**Hypothesis:** After Wiener decomposition, the specific components of formerly coherent bilateral pairs should have near-zero inter-channel coherence.

**Metric:** `reduction = coh_pre - coh_post` for all channel pairs. Positive reduction = coherence was removed. Passing criterion: `coh_post < 0.05` for bilateral pairs.

### V2 — Transitivity Constraint

**Hypothesis:** If channels i, j, k all receive signal from a single physical point source, the Wiener filter coefficients must satisfy:

```
|h_ij(f)| · |h_jk(f)| ≈ |h_ik(f)|          (amplitude transitivity)
∠h_ij(f) + ∠h_jk(f) ≈ ∠h_ik(f)  (mod 2π)  (phase transitivity)
```

**Metric:** `eps_amp` and `eps_phase` per channel triplet. Passing criteria: `eps_amp < 0.1`, `eps_phase < π/8 rad (0.392)`.

### V3 — Frequency Variation

**Hypothesis:** The physical medium (tissue, skull) imposes frequency-dependent attenuation. A truly frequency-dependent model should show significant variation in `|h(f)|` across the band.

**Metric:** `freq_variation = (max - min) / mean` of `|h(f)|` within the analysis band. Values > 20% support the frequency-dependent model over the scalar ablation baseline.

---

## Visualization

All visualization functions return `plt.Figure` objects and never call `plt.show()`. The caller is responsible for saving or displaying.

```python
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.wiener import decompose_epoch
from eeg_bg.visualization.filter_plots import plot_all_pairs_response
from eeg_bg.visualization.coherence_plots import (
    plot_coherence_matrix, plot_signal_decomposition
)
from eeg_bg.visualization.verification_plots import (
    plot_v2_transitivity, plot_v3_frequency_variation,
    plot_ica_vs_wiener_coherence
)
from eeg_bg.visualization.waveform_plots import plot_multichannel_comparison
import numpy as np

cfg = load_config()

# Load a cached epoch
data = np.load("cache/epochs/aaaaabhz/abc123.npz", allow_pickle=True)
epoch = data["epochs"][0]
ch_names = list(data["ch_names"])

# Decompose
result = decompose_epoch(epoch, ch_names, cfg, subject_id="aaaaabhz", epoch_idx=0)

# Filter frequency response for all bilateral pairs
fig = plot_all_pairs_response(result, save_path="results/filter_response.png")

# Raw / coherent / specific signal traces for FP1
fig = plot_signal_decomposition(result, ch_name="FP1", time_window=(0, 4))
fig.savefig("results/signal_decomp_FP1.png", dpi=150)

# Coherence matrix before/after
from eeg_bg.verification.coherence import compute_pairwise_coherence
coh_pre  = compute_pairwise_coherence(result.raw,      cfg["preprocessing"]["target_sfreq"],
                                       cfg["wiener"]["nperseg"], cfg["wiener"]["freq_band"])
coh_post = compute_pairwise_coherence(result.specific, cfg["preprocessing"]["target_sfreq"],
                                       cfg["wiener"]["nperseg"], cfg["wiener"]["freq_band"])
fig = plot_coherence_matrix(coh_pre, coh_post, ch_names, title="Subject aaaaabhz, Epoch 0")
fig.savefig("results/coherence_matrix.png", dpi=150)

# V2/V3 from verification CSVs
import pandas as pd
v2_df = pd.read_csv("results/v2_transitivity.csv")
v3_df = pd.read_csv("results/v3_frequency_variation.csv")
plot_v2_transitivity(v2_df).savefig("results/v2_transitivity.png", dpi=150)
plot_v3_frequency_variation(v3_df).savefig("results/v3_freq_variation.png", dpi=150)

# Stacked 19-channel waveform comparison (Raw | Wiener specific | ICA cleaned)
# Load arrays directly from caches — no re-decomposition needed
import numpy as np
edata = np.load("cache/epochs/aaaaabhz/abc123.npz", allow_pickle=True)
raw   = edata["epochs"][0]                                   # (19, 1000)
ch_names = list(edata["ch_names"])

wdata          = np.load("cache/wiener_frequency/aaaaabhz/abc123.npz", allow_pickle=True)
wiener_specific = wdata["specific"][0]                        # (19, 1000)

idata        = np.load("cache/ica/aaaaabhz/abc123.npz", allow_pickle=True)
ica_specific = idata["specific"][0]                           # (19, 1000)

fig = plot_multichannel_comparison(
    raw, wiener_specific, ica_specific,
    ch_names, sfreq=125.0, title="aaaaabhz  epoch 0"
)
fig.savefig("results/figures/aaaaabhz/waveform_comparison.png", dpi=150, bbox_inches="tight")
```

---

## Testing

The test suite uses **synthetic data only** — no real EDF files required on any machine.

```bash
# Run all unit tests
conda run -n eeg_pipeline pytest tests/ -v

# Exclude integration tests (default for CI)
conda run -n eeg_pipeline pytest tests/ -m "not integration" -v

# Run a specific module
conda run -n eeg_pipeline pytest tests/test_decomposition/ -v

# Quiet summary
conda run -n eeg_pipeline pytest tests/ -m "not integration" -q
```

**Current status:** 155 tests (excluding 1 integration test), all passing.

| Test module | Count | What is verified |
|-------------|-------|-----------------|
| `test_config.py` | 3 | YAML loading, path resolution |
| `test_io/` | 17 | Dataset traversal, EDF channel normalisation, annotation parsing (incl. full-recording semantics), cache read/write |
| `test_preprocessing/` | 9 | Epoch slicing, artifact rejection, bandpass filter, reference detection |
| `test_decomposition/` | 13 | Wiener decomposition identity (`specific + coherent = raw`), scalar ablation, ICA shape |
| `test_verification/` | 5 | V1 coherence reduction, V2/V3 DataFrame structure |
| `test_visualization/` | 4 | PSD figure shape, 3-column grid layout, config channel defaults |
| `test_features/` | 72 | Band power normalisation, Hjorth correctness, spectral entropy range, FEATURE_NAMES length/uniqueness, build_dataset split filtering, plus the disconnected asymmetry/wavelet/connectivity/complexity/temporal_stats modules' own unit tests |
| `test_ml/` | 31 | subject_level_predict aggregation, evaluate_subject_level keys, SHAP shape/non-negativity, plot functions create files, EEGNet/EEGEpochDataset/train_cnn smoke tests |
| `test_scripts/` | 1 | Script 06's stale-feature-cache shape guard |

CNN tests (`tests/test_ml/test_cnn_*.py`) are known to segfault on some macOS/torch environments; exclude with `-k "not cnn"` if that happens on your machine.

### Synthetic Fixture Design

`tests/conftest.py` generates a known point-source model:

```
x_i(t) = gain_i · source(t) + noise_i(t)
```

- `source(t)`: broadband Gaussian noise, σ = 50 µV
- `gains`: uniform random in [0.5, 1.0], seed-controlled
- `noise`: independent Gaussian per channel, σ = 1 µV (SNR ≈ 50 dB)
- `nperseg = 500` in `BASE_CFG` (2 segments per epoch) to ensure multi-segment Welch estimation for coherence tests

At this SNR, the Wiener filter should recover `gain_i / gain_j ≈ h_ij` and reduce cross-channel coherence of the specific component to near zero.

**Integration tests** (marked `@pytest.mark.integration`) require actual TUEP EDF files and are excluded from normal runs with `-m "not integration"`.

---

## Package API

See [`eeg_bg/README.md`](eeg_bg/README.md) for the full module-level API reference.
