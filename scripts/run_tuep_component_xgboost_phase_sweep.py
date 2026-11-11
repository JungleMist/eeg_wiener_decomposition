#!/usr/bin/env python3
"""Run four TUEP ECMAD phase thresholds through XGBoost and test-set SHAP."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import mkdtemp

import yaml

from eeg_bg.config.settings import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# docs/tuep_component_eegnet_xgboost_phase_descriptive_report_2026-09-18.html
PHASE_CONFIGS = {
    "0.01": "tuep_phase001.yaml",
    "0.05": "tuep_phase005.yaml",
    "0.1": "tuep_phase01.yaml",
    "0.5": "tuep_phase05.yaml",
    "1": "tuep_phase1.yaml",
}
DEFAULT_PHASES = ("0.01", "0.05", "0.1", "0.5")
FEATURE_SET = "base211_conn80"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases", nargs=4, choices=PHASE_CONFIGS, default=DEFAULT_PHASES,
        help=f"four distinct thresholds in radians (default: {' '.join(DEFAULT_PHASES)})",
    )
    parser.add_argument("--data-dir", type=Path, help="TUEP v3.1.0 root (defaults to the original config)")
    parser.add_argument("--cache-dir", type=Path, help="parent for temporary caches (defaults to the original config)")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "tuep_component_xgboost_phase_sweep",
        help="parent for the new run directory",
    )
    parser.add_argument("--workers", type=int, default=10, help="preprocessing and feature workers (default: 10)")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="XGBoost device (defaults to the original config)")
    args = parser.parse_args()
    if len(set(args.phases)) != 4:
        parser.error("--phases must contain four distinct thresholds")
    if args.workers < 1:
        parser.error("--workers must be positive")

    configs = {
        phase: load_config(PROJECT_ROOT / "configs" / PHASE_CONFIGS[phase])
        for phase in args.phases
    }
    original_paths = configs[args.phases[0]]["paths"]
    data_dir = (args.data_dir or Path(original_paths["data_root"])).expanduser().resolve()
    if not data_dir.is_dir():
        parser.error(f"TUEP data directory does not exist: {data_dir}. Set --data-dir.")
    cache_dir = (args.cache_dir or Path(original_paths["cache_dir"])).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(mkdtemp(prefix="run-", dir=output_dir))
    cache_run = Path(mkdtemp(prefix="tuep-xgb-sweep-", dir=cache_dir))
    print(f"Results: {run_dir}\nWorking cache: {cache_run}", flush=True)

    rows = []
    for phase, cfg in configs.items():
        phase_dir = run_dir / f"phase_{phase}"
        phase_dir.mkdir()
        phase_cache = cache_run / f"phase_{phase}"
        cfg["paths"].update(
            data_root=str(data_dir), cache_dir=str(phase_cache), results_dir=str(phase_dir),
        )
        if args.device is not None:
            cfg["ml"]["xgboost"]["fixed"]["device"] = args.device
        config_path = phase_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        common_args = ["--config", str(config_path), "--workers", str(args.workers)]
        print(f"ECMAD phase threshold: {phase} rad", flush=True)
        for script, extra_args in (
            ("21_cache_tuep_continuous_wiener.py", []),
            ("22_extract_tuep_component_epochs.py", []),
            (
                "24_train_tuep_component_xgboost.py",
                ["--condition", "components", "--feature-set", FEATURE_SET],
            ),
        ):
            command = [sys.executable, str(PROJECT_ROOT / "scripts" / script)]
            print(f"Running {script}", flush=True)
            subprocess.run([*command, *common_args, *extra_args], cwd=PROJECT_ROOT, check=True)

        summary = phase_dir / "tuep_component_xgboost" / FEATURE_SET / "comparison_summary.csv"
        with summary.open(encoding="utf-8", newline="") as handle:
            rows.extend({"phase_rad": phase, **row} for row in csv.DictReader(handle))
        shutil.rmtree(phase_cache)

    cache_run.rmdir()
    summary = run_dir / "comparison_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Sweep summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
