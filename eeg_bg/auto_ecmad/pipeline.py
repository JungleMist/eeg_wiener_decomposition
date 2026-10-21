"""Persistent stage orchestration with explicit subject-access boundaries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from importlib.metadata import version
import inspect
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from eeg_bg.config.settings import load_config
from .adapters import ERPCoreAdapter
from .artifacts import (SCHEMA_VERSION, atomic_write, code_fingerprint, fingerprint,
                        load_recording, read_json, save_recording, sha256_file,
                        write_csv, write_json)
from .decomposition import ECMADPairDecomposer, concatenate_batches
from .search import exhaustive_search
from .training import (EEGNetTrainer, class_ratio, evaluate_predictions, prediction_frame,
                       select_balanced_accuracy_threshold, subject_equal_bce,
                       validate_probabilities)
from .types import DatasetAdapter, FitResult, IneligibleRecording, ModelTrainer, PairDecomposer, TrialBatch

logger = logging.getLogger(__name__)


def split_subjects(subject_ids, options, seed):
    def split(ids, fraction, split_seed):
        if not 0 < fraction < 1 or len(ids) < 2:
            raise ValueError("Subject split requires >=2 subjects and a fraction in (0,1)")
        ids = np.asarray(sorted(set(ids)))
        shuffled = ids[np.random.default_rng(split_seed).permutation(len(ids))]
        size = min(len(ids) - 1, max(1, math.ceil(len(ids) * fraction)))
        return sorted(shuffled[size:].tolist()), sorted(shuffled[:size].tolist())

    ids = sorted(set(subject_ids))
    if len(ids) < 5:
        raise ValueError("At least five eligible subjects are required")
    pool, test = split(ids, float(options.get("test_size", .2)), seed)
    train, validation = split(pool, float(options.get("validation_size", .2)), seed + 1_000_003)
    inner_fit, inner_selection = split(train, float(options.get("selection_size", .2)), seed + 2_000_006)
    partitions = {"inner_fit": inner_fit, "inner_selection": inner_selection,
                  "validation": validation, "test": test}
    combined = [s for values in partitions.values() for s in values]
    if len(combined) != len(set(combined)) or set(combined) != set(ids):
        raise RuntimeError("Subject split is incomplete or leaking")
    return partitions


class AutoECMADPipeline:
    def __init__(self, config: dict | str | Path, *, output_dir: str | Path | None = None,
                 adapter: DatasetAdapter | None = None, decomposer: PairDecomposer | None = None,
                 trainer: ModelTrainer | None = None, force: bool = False):
        self.cfg = deepcopy(config) if isinstance(config, dict) else load_config(config)
        self.options = self.cfg["auto_ecmad"]
        if self.options["search"].get("input_mode") != "target_components_only":
            raise ValueError("Only target_components_only input is implemented")
        if self.options["search"].get("strategy") != "exhaustive":
            raise ValueError("Only exhaustive search is implemented")
        tolerance = float(self.options["search"].get("tie_tolerance", 1e-6))
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Tie tolerance must be nonnegative")
        if int(self.options.get("workers", 1)) < 1:
            raise ValueError("workers must be positive")
        if adapter is None and self.options["adapter"].get("name") != "erp_core":
            raise ValueError("Pass an adapter instance for custom datasets")
        if trainer is None and self.options["training"].get("backend") != "eegnet":
            raise ValueError("Pass a trainer instance for custom models")
        self.adapter = adapter if adapter is not None else ERPCoreAdapter(self.cfg)
        self.decomposer = decomposer if decomposer is not None else ECMADPairDecomposer(self.cfg)
        self.trainer = trainer if trainer is not None else EEGNetTrainer(self.options["training"])
        self.output_dir = Path(output_dir or self.cfg["paths"]["results_dir"]).expanduser().resolve()
        self.seed = int(self.options.get("random_state", 42))
        self.force = force
        self._state = None

    def _identity(self):
        backends = []
        for backend in (self.adapter, self.decomposer, self.trainer):
            cls = type(backend)
            try:
                implementation = fingerprint(inspect.getsource(cls))
            except (OSError, TypeError):
                implementation = "unavailable"
            backends.append({"type": f"{cls.__module__}.{cls.__qualname__}", "implementation": implementation})
        return {"schema": SCHEMA_VERSION, "config": self.cfg, "code": code_fingerprint(),
                "backends": backends,
                "runtime": {name: version(name) for name in ("numpy", "scipy", "mne", "torch", "scikit-learn")}}

    def prepare(self):
        """Prepare raw and split metadata only; no candidate decomposition."""
        state_path = self.output_dir / "prepared.json"
        if state_path.exists() and not self.force:
            return self._load_state()
        sources = self.adapter.discover()
        if len({s.recording_id for s in sources}) != len(sources):
            raise ValueError("Recording IDs must be globally unique")
        source_rows = []
        for source in sources:
            for path in source.files:
                path = path.resolve()
                stat = path.stat()
                source_rows.append({"path": str(path), "size": stat.st_size,
                                    "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path)})
        run_id = fingerprint({**self._identity(), "sources": source_rows})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "resolved_config.json", self.cfg)

        def prepare_source(item):
            index, source = item
            logger.info("Prepare %s", source.subject_id)
            try:
                record = self.adapter.prepare(source)
                record.validate()
                if record.y is None:
                    raise ValueError("Experiment preparation requires labels")
                relative = f"prepared/record_{index:04d}.npz"
                save_recording(self.output_dir / relative, record)
                return record, {"subject_id": record.subject_id, "recording_id": record.recording_id,
                                "cache": relative, "sha256": sha256_file(self.output_dir / relative),
                                "channels": list(record.channel_names), "sfreq": record.sfreq,
                                "offsets": record.offsets.tolist()}, None
            except IneligibleRecording as exc:
                return None, None, {"subject_id": source.subject_id,
                                    "recording_id": source.recording_id, "reason": str(exc)}

        entries, excluded, manifests = [], [], []
        # Parallelism is restricted to raw preparation; all fits remain serial.
        with threadpool_limits(limits=1):
            workers = int(self.options.get("workers", 1))
            if workers == 1:
                results = map(prepare_source, enumerate(sources))
                for record, entry, error in results:
                    self._collect_prepared(record, entry, error, entries, excluded, manifests)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for record, entry, error in pool.map(prepare_source, enumerate(sources)):
                        self._collect_prepared(record, entry, error, entries, excluded, manifests)
        write_json(self.output_dir / "excluded_recordings.json", excluded)
        if not entries:
            raise ValueError("No eligible recordings; see excluded_recordings.json")
        channels = tuple(entries[0]["channels"])
        if len(channels) < 2 or any(tuple(e["channels"]) != channels or e["sfreq"] != entries[0]["sfreq"]
                                    or e["offsets"] != entries[0]["offsets"] for e in entries):
            raise ValueError("Prepared recordings need identical channel order, sampling rate and epoch grid")
        partitions = split_subjects([e["subject_id"] for e in entries], self.options.get("splits", {}), self.seed)
        subject_split = {s: split for split, subjects in partitions.items() for s in subjects}
        manifest = pd.concat(manifests, ignore_index=True)
        manifest["split"] = manifest.subject_id.map(subject_split)
        if manifest.trial_id.duplicated().any():
            raise ValueError("Trial IDs must be globally unique")
        for split in partitions:
            if manifest.loc[manifest.split == split, "true_label"].nunique() != 2:
                raise ValueError(f"Partition {split} must contain both classes; use another explicit split seed")
        write_csv(self.output_dir / "trial_manifest.csv", manifest)
        write_csv(self.output_dir / "split_manifest.csv", pd.DataFrame([
            {"subject_id": s, "split": split} for split, subjects in partitions.items() for s in subjects]))
        state = {"fingerprint": run_id, "identity": self._identity(), "sources": source_rows,
                 "records": entries, "channels": list(channels), "partitions": partitions}
        write_json(state_path, state)  # Commit last: incomplete preparation cannot be resumed as complete.
        self._state = state
        return state

    @staticmethod
    def _collect_prepared(record, entry, error, entries, excluded, manifests):
        if error:
            excluded.append(error)
            logger.warning("Exclude %s: %s", error["subject_id"], error["reason"])
            return
        entries.append(entry)
        manifests.append(pd.DataFrame({"subject_id": record.subject_id, "recording_id": record.recording_id,
                                      "trial_id": record.trial_ids, "sample": record.samples, "true_label": record.y}))

    def _load_state(self):
        if self._state is not None:
            return self._state
        path = self.output_dir / "prepared.json"
        if not path.exists():
            raise ValueError("Run prepare before this stage")
        state = read_json(path)
        if fingerprint(state["identity"]) != fingerprint(self._identity()):
            raise ValueError("Configuration, code or backend changed; run prepare --force")
        discovered = self.adapter.discover()
        paths = {str(path.resolve()) for source in discovered for path in source.files}
        if paths != {row["path"] for row in state["sources"]}:
            raise ValueError("Source inventory changed; run prepare --force")
        # Only metadata is inspected here: search never reads held-out EEG caches.
        for row in state["sources"]:
            stat = Path(row["path"]).stat()
            if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
                raise ValueError("Source changed; run prepare --force")
        expected = fingerprint({**state["identity"], "sources": state["sources"]})
        if state["fingerprint"] != expected:
            raise ValueError("Invalid prepared run fingerprint")
        self._state = state
        return state

    def _epoch(self, record, data, channels):
        batch = self.adapter.epoch(record, data, channels)
        batch.validate()
        if batch.channel_names != channels or not np.array_equal(batch.trial_ids, record.trial_ids) or not np.all(
            batch.subject_ids == record.subject_id
        ) or ((batch.y is None) != (record.y is None)) or (
            record.y is not None and not np.array_equal(batch.y, record.y)
        ):
            raise ValueError("Adapter changed shared trial selection, labels or channel order")
        return batch

    def _record_batch(self, record, *, pair=None, mapping=None):
        diagnostics = []
        if pair is None and mapping is None:
            return self._epoch(record, record.data_uv, record.channel_names), diagnostics
        pairs = [pair] if pair else [(target, mapping[target]) for target in record.channel_names]
        batches = []
        for target, reference in pairs:
            if target == reference or target not in record.channel_names or reference not in record.channel_names:
                raise ValueError("Invalid directed reference mapping")
            result = self.decomposer.decompose(record.data_uv[record.channel_names.index(target)],
                                               record.data_uv[record.channel_names.index(reference)], record.sfreq)
            data = np.asarray([result.coherent, result.specific])
            if data.shape != (2, record.data_uv.shape[1]) or not np.isfinite(data).all():
                raise ValueError("Invalid pair decomposition output")
            if not np.allclose(data.sum(axis=0), record.data_uv[record.channel_names.index(target)], rtol=1e-7, atol=1e-7):
                raise ValueError("ECMAD reconstruction failed")
            # Candidate names intentionally do not encode target/ref: every fit
            # has exactly the same two-channel input contract and initialization.
            names = ("coherent", "specific") if pair else (f"{target}::coherent", f"{target}::specific")
            batches.append(self._epoch(record, data, names))
            diagnostics.append({"recording_id": record.recording_id, "target": target,
                                "reference": reference, **result.diagnostics})
        first = batches[0]
        return TrialBatch(np.concatenate([b.X for b in batches], axis=1), first.subject_ids,
                          first.trial_ids, tuple(ch for b in batches for ch in b.channel_names), first.y), diagnostics

    def _batch(self, subjects, *, pair=None, mapping=None):
        state = self._load_state()
        batches, diagnostics = [], []
        for entry in state["records"]:
            if entry["subject_id"] not in subjects:
                continue
            path = self.output_dir / entry["cache"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError("Prepared cache changed; run prepare --force")
            record = load_recording(path)
            batch, details = self._record_batch(record, pair=pair, mapping=mapping)
            batches.append(batch)
            diagnostics.extend(details)
        return concatenate_batches(batches), diagnostics

    def _fit_cached(self, out_dir, token, make_batches, metric):
        marker = out_dir / "complete.json"
        if marker.exists() and not self.force:
            saved = read_json(marker)
            if saved["fingerprint"] != token:
                raise ValueError("Training cache fingerprint differs; use --force")
            if any(not (out_dir / name).is_file() or sha256_file(out_dir / name) != digest
                   for name, digest in saved["files"].items()):
                raise ValueError("Incomplete/corrupt training cache; use --force")
            values = read_json(out_dir / "fit_result.json")
            return FitResult(out_dir / values["checkpoint"], values["best_epoch"], values["selection_score"],
                             np.asarray(values["validation_probabilities"]), values["metadata"])
        marker.unlink(missing_ok=True)
        train, validation, diagnostics = make_batches()
        train.validate(labeled=True)
        validation.validate(labeled=True)
        if set(train.subject_ids) & set(validation.subject_ids):
            raise ValueError("Subject leakage in training request")
        result = self.trainer.fit(train, validation, out_dir, selection_metric=metric, seed=self.seed)
        probabilities = validate_probabilities(result.validation_probabilities, len(validation.X))
        if metric == "bce":
            score = subject_equal_bce(validation, probabilities, class_ratio(train))
        else:
            score = float(average_precision_score(validation.y, probabilities))
        if not np.isclose(score, result.selection_score, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Trainer selection score differs from required {metric} objective")
        if not np.isfinite(result.selection_score):
            raise ValueError("Non-finite trainer score")
        relative = result.checkpoint.resolve().relative_to(out_dir.resolve())
        write_csv(out_dir / "val_predictions.csv", prediction_frame(validation, probabilities))
        write_json(out_dir / "decomposition_diagnostics.json", diagnostics)
        write_json(out_dir / "fit_result.json", {
            "checkpoint": str(relative), "best_epoch": result.best_epoch,
            "selection_score": result.selection_score, "validation_probabilities": probabilities,
            "metadata": result.metadata,
        })
        names = {str(relative), "fit_result.json", "val_predictions.csv", "decomposition_diagnostics.json"}
        if (out_dir / "history.csv").exists():
            names.add("history.csv")
        files = {name: sha256_file(out_dir / name) for name in names}
        write_json(marker, {"fingerprint": token, "files": files})
        return result

    def search(self):
        state = self._load_state()
        channels = tuple(state["channels"])
        partitions = state["partitions"]

        def fit_candidate(target, reference, out_dir):
            def make_batches():
                train, td = self._batch(partitions["inner_fit"], pair=(target, reference))
                selection, vd = self._batch(partitions["inner_selection"], pair=(target, reference))
                return train, selection, td + vd
            token = fingerprint({"run": state["fingerprint"], "target": target, "reference": reference,
                                 "input_mode": "target_components_only", "selection": "bce"})
            return self._fit_cached(out_dir, token, make_batches, "bce")

        _, selected = exhaustive_search(channels, self.output_dir, fit_candidate,
                                        float(self.options["search"].get("tie_tolerance", 1e-6)))
        mapping = {"run_fingerprint": state["fingerprint"], "channel_names": channels,
                   "references": {row["target"]: row["reference"] for row in selected},
                   "selected": selected, "decomposition": self.cfg["wiener"],
                   "selection": "inner_selection_subject_equal_class_weighted_bce"}
        mapping["fingerprint"] = fingerprint(mapping)
        write_json(self.output_dir / "selected_mapping.json", mapping)
        return mapping

    def _mapping(self):
        path = self.output_dir / "selected_mapping.json"
        if not path.exists():
            raise ValueError("Run search to freeze the reference mapping first")
        mapping = read_json(path)
        token = mapping.pop("fingerprint")
        if token != fingerprint(mapping) or mapping["run_fingerprint"] != self._load_state()["fingerprint"]:
            raise ValueError("Reference mapping fingerprint mismatch; rerun search")
        mapping["fingerprint"] = token
        if mapping["channel_names"] != self._load_state()["channels"] or set(mapping["references"]) != set(mapping["channel_names"]):
            raise ValueError("Reference mapping channel mismatch")
        return mapping

    def fit(self):
        state, mapping = self._load_state(), self._mapping()
        partitions = state["partitions"]
        for condition in ("selected", "raw"):
            references = mapping["references"] if condition == "selected" else None

            def make_batches():
                train, td = self._batch(partitions["inner_fit"] + partitions["inner_selection"], mapping=references)
                validation, vd = self._batch(partitions["validation"], mapping=references)
                return train, validation, td + vd

            token = fingerprint({"run": state["fingerprint"], "mapping": mapping["fingerprint"],
                                 "condition": condition, "selection": "auprc"})
            directory = self.output_dir / "models" / condition
            result = self._fit_cached(directory, token, make_batches, "auprc")
            predictions = pd.read_csv(directory / "val_predictions.csv")
            threshold = select_balanced_accuracy_threshold(predictions.true_label.to_numpy(),
                                                           result.validation_probabilities)
            frozen = {
                "fingerprint": token, "mapping_fingerprint": mapping["fingerprint"],
                "checkpoint": str(result.checkpoint.relative_to(directory)),
                "checkpoint_sha256": sha256_file(result.checkpoint), "threshold": threshold,
                "best_epoch": result.best_epoch, "selection_score": result.selection_score,
            }
            frozen["freeze_fingerprint"] = fingerprint(frozen)
            write_json(directory / "frozen_model.json", frozen)
        self._export_bundle(mapping)
        return read_json(self.output_dir / "models/selected/frozen_model.json")

    def _export_bundle(self, mapping):
        directory = self.output_dir / "bundle"
        directory.mkdir(parents=True, exist_ok=True)
        model_dir = self.output_dir / "models/selected"
        frozen = read_json(model_dir / "frozen_model.json")
        atomic_write(directory / "model.pt", lambda handle: handle.write((model_dir / frozen["checkpoint"]).read_bytes()))
        write_json(directory / "config.json", self.cfg)
        write_json(directory / "mapping.json", mapping)
        write_json(directory / "model.json", frozen)
        write_json(directory / "manifest.json", {"schema": SCHEMA_VERSION,
                   "files": {name: sha256_file(directory / name) for name in
                             ("model.pt", "config.json", "mapping.json", "model.json")}})

    def evaluate(self):
        state, mapping = self._load_state(), self._mapping()
        frozen_models = {}
        # Both models must already be frozen before any held-out EEG is loaded.
        for condition in ("selected", "raw"):
            directory = self.output_dir / "models" / condition
            frozen = read_json(directory / "frozen_model.json")
            frozen_payload = {key: value for key, value in frozen.items() if key != "freeze_fingerprint"}
            expected = fingerprint({"run": state["fingerprint"], "mapping": mapping["fingerprint"],
                                    "condition": condition, "selection": "auprc"})
            if frozen["fingerprint"] != expected or frozen["freeze_fingerprint"] != fingerprint(frozen_payload) or sha256_file(directory / frozen["checkpoint"]) != frozen["checkpoint_sha256"]:
                raise ValueError("Frozen model differs from selected run")
            frozen_models[condition] = frozen
        summaries = {}
        for condition, frozen in frozen_models.items():
            directory = self.output_dir / "models" / condition
            summaries[condition] = {}
            for split in ("validation", "test"):
                batch, details = self._batch(state["partitions"][split], mapping=mapping["references"] if condition == "selected" else None)
                probabilities = self.trainer.predict(directory / frozen["checkpoint"], batch)
                metrics = evaluate_predictions(batch, probabilities, frozen["threshold"])
                write_csv(directory / f"{split}_predictions.csv", prediction_frame(batch, probabilities, frozen["threshold"]))
                write_json(directory / f"{split}_metrics.json", metrics)
                write_json(directory / f"{split}_decomposition_diagnostics.json", details)
                summaries[condition][split] = metrics
        write_json(self.output_dir / "summary.json", {"results": summaries,
                   "limitations": "Single seed/split; classification utility is not proof of physiological denoising quality."})
        lines = ["# Auto ECMAD ERN", "", "Single seed/split; classification utility does not establish physiological denoising quality.", ""]
        for condition, splits in summaries.items():
            metrics = splits["test"]["trial"]
            lines.append(f"- {condition}: test AUROC={metrics['auroc']:.6f}, AUPRC={metrics['auprc']:.6f}, F1={metrics['f1']:.6f}")
        atomic_write(self.output_dir / "report.md", lambda handle: handle.write(("\n".join(lines) + "\n").encode()))
        return summaries

    @classmethod
    def from_bundle(cls, bundle_dir, *, adapter=None, decomposer=None, trainer=None,
                    data_dir=None, device=None, output_dir=None):
        bundle_dir = Path(bundle_dir).resolve()
        manifest = read_json(bundle_dir / "manifest.json")
        if manifest["schema"] != SCHEMA_VERSION or any(sha256_file(bundle_dir / name) != digest
                                                      for name, digest in manifest["files"].items()):
            raise ValueError("Invalid inference bundle")
        cfg = read_json(bundle_dir / "config.json")
        if data_dir:
            cfg["auto_ecmad"]["adapter"]["data_dir"] = str(data_dir)
            cfg["auto_ecmad"]["adapter"]["subjects"] = None
        if device:
            cfg["auto_ecmad"]["training"]["device"] = device
        result = cls(cfg, output_dir=output_dir or bundle_dir.parent / "predictions",
                     adapter=adapter, decomposer=decomposer, trainer=trainer)
        result._bundle = bundle_dir
        return result

    def predict(self, records=None):
        """Use from_bundle() for standalone inference, including unlabeled inputs."""
        bundle = getattr(self, "_bundle", self.output_dir / "bundle")
        manifest = read_json(bundle / "manifest.json")
        if any(sha256_file(bundle / name) != digest for name, digest in manifest["files"].items()):
            raise ValueError("Inference bundle checksum mismatch")
        mapping = read_json(bundle / "mapping.json")
        frozen = read_json(bundle / "model.json")
        saved_cfg = read_json(bundle / "config.json")
        if fingerprint({key: saved_cfg[key] for key in ("wiener", "preprocessing")}) != fingerprint({
            key: self.cfg[key] for key in ("wiener", "preprocessing")
        }) or saved_cfg["erp_core"]["ern"] != self.cfg["erp_core"]["ern"]:
            raise ValueError("Inference preprocessing differs from bundle; use from_bundle")
        if records is None:
            records = (self.adapter.prepare(source, inference=True) for source in self.adapter.discover())
        frames = []
        for record in records:
            if tuple(mapping["channel_names"]) != record.channel_names:
                raise ValueError("Inference channel order differs from frozen mapping")
            if not np.isclose(record.sfreq, self.cfg["preprocessing"]["target_sfreq"]):
                raise ValueError("Inference sampling rate differs from frozen preprocessing")
            batch, _ = self._record_batch(record, mapping=mapping["references"])
            probabilities = self.trainer.predict(bundle / "model.pt", batch)
            frames.append(prediction_frame(batch, probabilities, frozen["threshold"]))
        if not frames:
            raise ValueError("No prediction trials")
        frame = pd.concat(frames, ignore_index=True)
        write_csv(self.output_dir / "predictions.csv", frame)
        return frame

    def run(self):
        self.prepare()
        self.search()
        self.fit()
        return self.evaluate()
