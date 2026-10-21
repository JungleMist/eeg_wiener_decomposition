"""Exhaustive directed-pair search, independent of any dataset or model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .artifacts import write_csv
from .types import FitResult

logger = logging.getLogger(__name__)


def select_reference(rows: list[dict], channels: tuple[str, ...], tolerance: float = 1e-6) -> dict:
    if not rows or not all(np.isfinite(row["selection_loss"]) for row in rows):
        raise ValueError("All candidate losses must be finite")
    minimum = min(row["selection_loss"] for row in rows)
    eligible = [row for row in rows if row["selection_loss"] <= minimum + tolerance]
    chosen = min(eligible, key=lambda row: channels.index(row["reference"]))
    ordered = sorted(rows, key=lambda row: (row["selection_loss"], channels.index(row["reference"])))
    return {**chosen, "runner_up_gap": float(ordered[1]["selection_loss"] - ordered[0]["selection_loss"])
            if len(ordered) > 1 else None}


def exhaustive_search(channels: tuple[str, ...], output_dir: Path,
                      fit_candidate: Callable[[str, str, Path], FitResult],
                      tolerance: float = 1e-6) -> tuple[list[dict], list[dict]]:
    all_rows, selected = [], []
    for t, target in enumerate(channels):
        rows = []
        for i, reference in enumerate(channels):
            if i == t:
                continue
            candidate_id = f"t{t:03d}_r{i:03d}"
            logger.info("Candidate %d/%d: %s <- %s", len(all_rows) + 1,
                        len(channels) * (len(channels) - 1), target, reference)
            result = fit_candidate(target, reference, output_dir / "candidates" / candidate_id)
            row = {"target": target, "reference": reference, "candidate_id": candidate_id,
                   "selection_loss": float(result.selection_score), "best_epoch": result.best_epoch}
            rows.append(row)
            all_rows.append(row)
            write_csv(output_dir / "candidate_scores.csv", pd.DataFrame(all_rows))
        if len(rows) != len(channels) - 1:
            raise RuntimeError("Cannot choose a reference from incomplete search")
        selected.append(select_reference(rows, channels, tolerance))
    return all_rows, selected
