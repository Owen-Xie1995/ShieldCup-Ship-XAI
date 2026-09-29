"""Record the exact Mac sources and read-only checkpoints used in Stage 22."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "weights": (282678575, "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473"),
    "official_generator": (None, "3f0c8830e590f6ad46934c034a5bf75f2d2b1d89baf2dba6683cdc1d9dc61df1"),
    "official_scorer": (None, "5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda"),
    "prior_control_csv": (None, "2e177fff03991c529767393b4aceb796cc380d7a724d6c0e31d5fbe829cfe196"),
    "prior_checkpoint_zip": (None, "66f7b405136f6cbb1a17c00c83ccff0d9a2801aeadc3fef58a79bb19af9850cb"),
    "stage20_checkpoint_zip": (10083094, "416927931fa74d5c0deec4282e333c03d124eeadf7dead0b6eccde8d9dd76dc2"),
    "stage20_experiment": (None, "3cb6ec81c0214012ff85833020a0f2fddee175b452c492412129a470cdcd08fd"),
    "stage21_checkpoint_zip": (510091, "9b36d2b568019da9809ef534d181f266dace0f042ede37ecdf0dbdfae57a7fee"),
    "stage21_experiment": (None, "5264364073961e5c3f8de1e1982fe2856de4eeecca40aca4ca5ca459784360f5"),
    "stage21_csv": (None, "c63544390619d5b83b85d2c81ac356985364c9235c254998ac34efb0eae3d267"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage17-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, default=ROOT.parent / "output_mac_next")
    parser.add_argument("--stage20-root", type=Path, default=ROOT.parent / "output_mac_stage20")
    parser.add_argument("--stage21-root", type=Path, default=ROOT.parent / "output_mac_stage21")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    stage17 = args.stage17_root.resolve()
    official = stage17 / "official"
    prior = args.prior_root.resolve()
    stage20 = args.stage20_root.resolve()
    stage21 = args.stage21_root.resolve()
    output = args.output_dir.resolve()
    quantus = Path(importlib.metadata.distribution("quantus").locate_file("quantus"))
    files = {
        "weights": args.weights.resolve(),
        "stage17_quick_eval": stage17 / "quick_eval.py",
        "official_wheel": official / "dist/kaggle-0.1.0-py3-none-any.whl",
        "official_generator": official / "vendor/kaggle/submmit/submmit.py",
        "official_scorer": official / "script/metrix.py",
        "official_score_wrapper": official / "score_c3.py",
        "official_test_csv": official / "data/test.csv",
        "official_solution": official / "script/solution_all_public.csv",
        "quantus_baseline_utils": quantus / "helpers/utils.py",
        "quantus_perturb_func": quantus / "functions/perturb_func.py",
        "quantus_faithfulness": quantus / "metrics/faithfulness/faithfulness_correlation.py",
        "prior_control_csv": prior / "runs/control.csv",
        "prior_control_run": prior / "runs/control.run.json",
        "prior_control_experiment": prior / "experiment.py",
        "prior_checkpoint_zip": prior.parent / "output_mac_next.zip",
        "stage20_verification": stage20 / "verification.json",
        "stage20_summary": stage20 / "summary.json",
        "stage20_experiment": stage20 / "experiment.py",
        "stage20_checkpoint_zip": stage20.parent / "output_mac_stage20.zip",
        "stage21_checkpoint_zip": stage21.parent / "output_mac_stage21.zip",
        "stage21_experiment": stage21 / "experiment.py",
        "stage21_csv": stage21 / "runs/black_occlusion16_signed.csv",
        "stage21_run": stage21 / "runs/black_occlusion16_signed.run.json",
        "stage21_score": stage21 / "runs/black_occlusion16_signed.score.json",
        "stage21_cache_config": stage21 / "cache/black_occlusion16_signed/config.json",
        "stage21_verification": stage21 / "verification.json",
        "stage21_summary": stage21 / "summary.json",
        "stage22_instruction": Path("/Users/xiefujun/Downloads/給Mac本地Codex_第22階段8x8遮蔽與熱區驗證指令.md"),
    }
    for method in ("black_grad_signed", "black_grad_abs", "black_ig50_mean5"):
        files[f"stage20_csv/{method}"] = stage20 / "runs" / f"{method}.csv"
    test = pd.read_csv(files["official_test_csv"])
    if test.columns.tolist() != ["image", "category"] or len(test) != 9 or test.image.duplicated().any():
        raise ValueError("Unexpected official test rows")
    for filename in test.image:
        files[f"test_image/{filename}"] = official / "data/test_images" / filename
    for path in sorted(output.glob("*.py")):
        files[f"stage22_code/{path.name}"] = path
    for path in sorted(output.glob("*.md")):
        files[f"stage22_doc/{path.name}"] = path
    for filename in ("probe.json", "probe.log"):
        files[f"stage22/{filename}"] = output / filename
    missing = [f"{name}: {path}" for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source files missing:\n" + "\n".join(missing))
    records = {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in files.items()
    }
    for name, (size, digest) in EXPECTED.items():
        record = records[name]
        if (size is not None and record["bytes"] != size) or record["sha256"] != digest:
            raise ValueError(f"Unexpected verified source {name}: {record['path']}")
    versions = {}
    for package in ("torch", "torchvision", "captum", "quantus", "numpy", "pandas", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        "python": sys.version.split()[0], "interpreter": sys.executable,
        "platform": platform.platform(), "packages": versions,
        "files": records,
        "test_order": test[["image", "category"]].to_dict(orient="records"),
        "scope": "nine organizer-supplied images; unseen platform result unknown",
    }
    atomic_json(output / "environment.json", payload)
    print(output / "environment.json")


if __name__ == "__main__":
    main()
