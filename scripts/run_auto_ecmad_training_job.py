#!/usr/bin/env python3
"""Supervise the full training/analysis chain and persist its job status."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/erp_core_auto_ecmad_eegnet.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "job.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("An auto ECMAD job already owns this output directory")
    status = {"state": "running", "pid": os.getpid(), "device": args.device,
              "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "config": str(Path(args.config).resolve()), "output_dir": str(output)}
    child = None

    def save(**updates):
        status.update(updates)
        status["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        temp = output / ".job_status.json.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(output / "job_status.json")

    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
        save(state="interrupted", signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    stages = (
        ("training", [sys.executable, "-u", str(root / "scripts/25_optimize_erp_core_ecmad_eegnet.py"),
                      "all", "--config", str(Path(args.config).resolve()),
                      "--output-dir", str(output), "--device", args.device]),
        ("analysis", [sys.executable, "-u", str(root / "scripts/analyze_auto_ecmad_results.py"),
                      "--results-dir", str(output)]),
    )
    try:
        for phase, command in stages:
            with (output / f"{phase}.log").open("a", buffering=1) as log:
                log.write(f"\n# Started {phase}: {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}\n")
                child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                save(phase=phase, command=command, child_pid=child.pid)
                rc = child.wait()
                if rc:
                    save(state="failed", return_code=rc)
                    return rc
        save(state="completed", return_code=0,
             report=str(output / "analysis/report.md"))
        return 0
    except Exception as exc:
        save(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
