#!/usr/bin/env python3
"""Run or resume two-stage, EEGNet-guided ERP-CORE ECMAD optimization."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eeg_bg.auto_ecmad import AutoECMADPipeline
from eeg_bg.config.settings import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all",
                        choices=("prepare", "search", "train", "evaluate", "predict", "all"))
    parser.add_argument("--config", default="configs/erp_core_auto_ecmad_eegnet.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--workers", type=int, help="Raw preparation threads, not GPU count")
    parser.add_argument("--bundle", type=Path, help="Standalone inference bundle (predict only)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.bundle and args.stage != "predict":
        parser.error("--bundle is only valid for predict")
    if args.stage == "predict" and args.bundle:
        pipeline = AutoECMADPipeline.from_bundle(args.bundle, data_dir=args.data_dir,
                                                device=args.device, output_dir=args.output_dir)
    else:
        cfg = load_config(args.config)
        if args.data_dir:
            cfg["auto_ecmad"]["adapter"]["data_dir"] = str(args.data_dir.resolve())
        if args.device:
            cfg["auto_ecmad"]["training"]["device"] = args.device
        if args.workers is not None:
            cfg["auto_ecmad"]["workers"] = args.workers
        pipeline = AutoECMADPipeline(cfg, output_dir=args.output_dir, force=args.force)
    method = {"all": "run", "train": "fit"}.get(args.stage, args.stage)
    getattr(pipeline, method)()


if __name__ == "__main__":
    main()
