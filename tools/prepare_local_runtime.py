"""Verify and copy locally supplied competition assets into the ignored runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    ("checkpoints/best_resnet50.pth", "checkpoints/best_resnet50.pth", "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473"),
    ("dist/kaggle-0.1.0-py3-none-any.whl", "dist/kaggle-0.1.0-py3-none-any.whl", "43986f644c383f255e73dcbd2305e311f15fd70b4aeec54deb346dab5d5cbfd7"),
    ("script/metrix.py", "src/official_metrix.py", "5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda"),
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Verify assets without copying")
    args = parser.parse_args()
    verified = []
    for source_name, target_name, expected in ASSETS:
        source, target = args.official_dir / source_name, ROOT / "runtime" / target_name
        if not source.is_file() or digest(source) != expected:
            raise RuntimeError(f"Missing or mismatched official asset: {source_name}")
        if target.exists() and (not target.is_file() or digest(target) != expected):
            raise FileExistsError(f"Refusing to replace a different runtime asset: {target_name}")
        verified.append((source, target, expected))
    for source, target, expected in verified:
        if not args.dry_run and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if digest(target) != expected:
                raise RuntimeError(f"Copy verification failed: {target.name}")
    print(json.dumps({"status": "PASS", "dry_run": args.dry_run,
                      "assets": [{"destination": str(target.relative_to(ROOT)), "sha256": expected}
                                 for _, target, expected in verified]}, indent=2))


if __name__ == "__main__":
    main()
