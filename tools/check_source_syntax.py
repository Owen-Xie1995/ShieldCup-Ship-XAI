"""Parse maintained Python code and archived snapshots without importing them."""
from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = sorted(path for name in ("src", "runtime", "tools", "experiments")
                   for path in (root / name).rglob("*.py")
                   if "__pycache__" not in path.parts and "stage23_public_source" not in path.parts
                   and path.name != "official_metrix.py")
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(root)))
    print(f"PASS: parsed {len(files)} Python sources without imports or model inference")


if __name__ == "__main__":
    main()
