"""Summarise predeclared paired seeds for the Stage 22 candidate and control.

Only run if the score and visual branch gate calls for paired repeats. This
script reads official-generator CSVs; it never edits or constructs them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = (314159, 271828)
METHOD = "black_occlusion8_signed"
COLUMNS = ["Id", "Faithfulness", "Max_Sensitivity", "Sparsity"]


def frame(path: Path) -> pd.DataFrame:
    value = pd.read_csv(path)
    if value.columns.tolist() != COLUMNS or value.Id.tolist() != list(range(1, 10)):
        raise ValueError(f"Invalid official CSV schema/order: {path}")
    if not np.isfinite(value[COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite metric: {path}")
    return value


def row_scores(value: pd.DataFrame) -> np.ndarray:
    return (20.0 * (1.0 + value.Faithfulness.to_numpy())
            + 30.0 * np.maximum(1.0 - value.Max_Sensitivity.to_numpy() / 2.0, 0.0)
            + 30.0 * value.Sparsity.to_numpy())


def atomic_write(path: Path, content: str) -> None:
    pending = path.with_name(path.name + ".pending")
    try:
        with pending.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    paired = args.output_dir / "paired_runs"
    rows = []
    summaries = []
    for seed in SEEDS:
        control_stem = f"control_seed{seed}"
        candidate_stem = f"{METHOD}_seed{seed}"
        control = frame(paired / f"{control_stem}.csv")
        candidate = frame(paired / f"{candidate_stem}.csv")
        cs = row_scores(control)
        ns = row_scores(candidate)
        delta = ns - cs
        for i in range(9):
            record = {"seed": seed, "Id": i + 1, "control_row_score": float(cs[i]),
                      "candidate_row_score": float(ns[i]), "delta_row_score": float(delta[i])}
            for metric in COLUMNS[1:]:
                record[f"control_{metric}"] = float(control.iloc[i][metric])
                record[f"candidate_{metric}"] = float(candidate.iloc[i][metric])
                record[f"delta_{metric}"] = record[f"candidate_{metric}"] - record[f"control_{metric}"]
            rows.append(record)
        summaries.append({
            "seed": seed,
            "control_score": float(cs.mean()),
            "candidate_score": float(ns.mean()),
            "delta_score": float(delta.mean()),
            "median_row_delta": float(np.median(delta)),
            "images_improved": int((delta > 0).sum()),
            "improved_ids": [int(i + 1) for i in np.flatnonzero(delta > 0)],
        })
    atomic_write(args.output_dir / "paired_comparison.csv", pd.DataFrame(rows).to_csv(index=False))
    result = {"scope": "organizer-supplied nine images only", "seeds": summaries,
              "method": METHOD, "control": "zero-baseline IG50"}
    atomic_write(args.output_dir / "paired_summary.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
