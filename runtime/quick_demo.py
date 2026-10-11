"""Run an analytic contract check using only a literal synthetic model/input."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from smoke import run_toy


def main() -> None:
    result = run_toy(torch.device("cpu"))
    print(json.dumps({"status": "PASS", "scope": "Synthetic CPU analytic check; no competition data or official model", **result}, indent=2))


if __name__ == "__main__":
    main()
