"""Recompute the two paired comparisons from the saved, nine-image CSV files."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (42, "experiments/stage25/results/reference/black_occlusion8_signed_mps_seed42.csv",
     "experiments/stage25/results/official9/black_occlusion4_soft15_mps_seed42_full9.csv"),
    (314159, "experiments/stage26/results/official9/black_occlusion8_signed_mps_seed314159_full9.csv",
     "experiments/stage26/results/official9/black_occlusion4_soft15_mps_seed314159_full9.csv"),
)


def read_scores(path: Path) -> list[float]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Id", "Faithfulness", "Max_Sensitivity", "Sparsity"]:
            raise ValueError(f"Unexpected columns: {path}")
        rows = list(reader)
    if [int(row["Id"]) for row in rows] != list(range(1, 10)):
        raise ValueError(f"Expected image IDs 1 through 9: {path}")
    scores = []
    for row in rows:
        faith, sensitivity, sparsity = (float(row[key]) for key in
                                        ("Faithfulness", "Max_Sensitivity", "Sparsity"))
        if not all(math.isfinite(value) for value in (faith, sensitivity, sparsity)):
            raise ValueError(f"Nonfinite metric: {path}")
        scores.append(20 * (1 + faith) + 30 * max(1 - sensitivity / 2, 0) + 30 * sparsity)
    return scores


def comparisons() -> list[dict]:
    results = []
    for seed, control_path, candidate_path in CASES:
        control = read_scores(ROOT / control_path)
        candidate = read_scores(ROOT / candidate_path)
        results.append({
            "seed": seed,
            "control": sum(control) / 9,
            "candidate": sum(candidate) / 9,
            "difference": sum(b - a for a, b in zip(control, candidate)) / 9,
            "images_improved": sum(b > a for a, b in zip(control, candidate)),
            "control_csv": control_path,
            "candidate_csv": candidate_path,
        })
    return results


def check_readme(results: list[dict]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for result in results:
        for value in (result["control"], result["candidate"]):
            if f"{value:.6f}" not in readme:
                raise ValueError(f"README score differs for seed {result['seed']}: {value:.6f}")
        if f"+{result['difference']:.6f}" not in readme:
            raise ValueError(f"README difference differs for seed {result['seed']}")
        if result["images_improved"] != 9:
            raise ValueError(f"Saved results no longer support 9/9 improvement for {result['seed']}")


if __name__ == "__main__":
    results = comparisons()
    check_readme(results)
    print(json.dumps({"status": "PASS", "scope": "Saved CSV arithmetic; no new model evaluation",
                      "comparisons": results}, indent=2))
