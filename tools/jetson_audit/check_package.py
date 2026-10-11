#!/usr/bin/env python3
"""Check a submission manifest, Python syntax and metric CSV without imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path, PurePosixPath


COLUMNS = ["Id", "Faithfulness", "Max_Sensitivity", "Sparsity"]
REQUIRED = (
    "submission.csv", "checkpoints/best_resnet50.pth", "run_submission.py",
    "custom_explain.py", "Dockerfile", "requirements-jetson.txt",
    "dist/kaggle-0.1.0-py3-none-any.whl", "script/metrix.py", "README.md",
)
CORE = (
    "custom_explain.py", "checkpoints/best_resnet50.pth", "submission.csv",
    "dist/kaggle-0.1.0-py3-none-any.whl", "script/metrix.py", "requirements-jetson.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(value: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("Manifest path must use forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) != value:
        raise ValueError("Unsafe or noncanonical manifest path")
    return value


def check_csv(path: Path, expected_rows: int) -> dict:
    errors = []
    result = {"passed": False, "errors": errors}
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        result.update(columns=columns, rows=len(rows))
        if columns != COLUMNS or len(rows) != expected_rows:
            errors.append("Unexpected CSV columns or row count")
        if not rows:
            errors.append("CSV must contain rows")
        if errors:
            return result
        # DictReader stores excess fields under None and fills missing fields
        # with None. An exact header alone does not establish each row's shape.
        if any(None in row or any(row[name] is None for name in COLUMNS) for row in rows):
            errors.append("CSV rows must have exactly four fields")
            return result
        ids = [float(row["Id"]) for row in rows]
        if ids != list(range(1, expected_rows + 1)):
            errors.append("IDs must be finite integers 1..N in order")
        values = {name: [float(row[name]) for row in rows] for name in COLUMNS[1:]}
        for name, numbers in values.items():
            if not all(math.isfinite(value) for value in numbers):
                errors.append("Non-finite metric: " + name)
        for name, low, high in (("Faithfulness", -1, 1), ("Max_Sensitivity", 0, math.inf), ("Sparsity", 0, 1)):
            if not all(low <= value <= high for value in values[name]):
                errors.append("Out-of-range metric: " + name)
        if errors:
            return result
        result["metrics"] = {name: {"min": min(numbers), "max": max(numbers), "mean": statistics.mean(numbers)}
                             for name, numbers in values.items()}
        result["attached_csv_formula_score"] = 100 * (
            0.4 * statistics.mean((value + 1) / 2 for value in values["Faithfulness"])
            + 0.3 * statistics.mean(max(1 - value / 2, 0) for value in values["Max_Sensitivity"])
            + 0.3 * statistics.mean(values["Sparsity"])
        )
        result["score_note"] = "Arithmetic from attached CSV; no model run or official evaluation."
        result["passed"] = True
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"CSV unreadable or malformed: {type(exc).__name__}: {exc}")
    return result


def audit(package: Path, expected_rows: int, baseline_manifest: Path | None = None,
          require_unchanged=()) -> dict:
    package = package.resolve()
    errors, checks = [], []
    manifest_file = package / "manifest.json"
    if manifest_file.is_symlink() or not manifest_file.resolve().is_relative_to(package):
        raise ValueError("Manifest must be a regular file within the package")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "sha256" or manifest.get("package_root") != package.name:
        errors.append("Manifest algorithm/package_root mismatch")
    recorded = []
    for entry in manifest["files"]:
        name = manifest_path(entry["path"])
        recorded.append(name)
        path = package / name
        check = {"path": name, "expected_size": entry["size"], "expected_sha256": entry["sha256"], "passed": False}
        # No dereferencing package links to read files outside the audit scope.
        if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(package):
            check.update(actual_size=path.stat().st_size, actual_sha256=sha256(path))
            check["passed"] = check["actual_size"] == entry["size"] and check["actual_sha256"] == entry["sha256"]
        if not check["passed"]:
            errors.append("Manifest mismatch or nonregular file: " + name)
        checks.append(check)
    inventory = list(package.rglob("*"))
    links = [str(path.relative_to(package)) for path in inventory if path.is_symlink()]
    if links:
        errors.append("Package contains symlinks: " + ", ".join(links))
    actual = {str(path.relative_to(package)) for path in inventory if path.is_file() or path.is_symlink()}
    expected = set(recorded) | {"manifest.json"}
    if len(recorded) != len(set(recorded)):
        errors.append("Duplicate manifest paths")
    if actual != expected:
        errors.append("Manifest coverage differs from actual inventory")
    required = {name: name in actual for name in REQUIRED}
    required["docker/<image>.tar"] = any(name.startswith("docker/") and name.endswith(".tar") for name in actual)
    if not all(required.values()):
        errors.append("Required deliverable missing")
    syntax = []
    for path in sorted(package.rglob("*.py")):
        name = str(path.relative_to(package))
        if path.is_symlink() or not path.resolve().is_relative_to(package):
            continue
        try:
            compile(path.read_bytes(), name, "exec")
            syntax.append({"path": name, "passed": True, "executed": False})
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append("Python syntax failed: " + name)
            syntax.append({"path": name, "passed": False, "executed": False, "error": str(exc)})
    csv_file = package / "submission.csv"
    csv_result = check_csv(csv_file, expected_rows) if not csv_file.is_symlink() and csv_file.resolve().is_relative_to(package) else {
        "passed": False, "errors": ["CSV must be a regular file within the package"]}
    if not csv_result["passed"]:
        errors.extend(csv_result["errors"])
    comparison = []
    if baseline_manifest:
        prior = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        if prior.get("algorithm") != "sha256":
            errors.append("Baseline manifest must use sha256")
        baseline = {manifest_path(entry["path"]): entry for entry in prior["files"]}
        for check in checks:
            previous = baseline.get(check["path"])
            unchanged = previous is not None and check.get("actual_size") == previous["size"] and check.get("actual_sha256") == previous["sha256"]
            comparison.append({"path": check["path"], "status": "unchanged" if unchanged else "changed_or_added"})
        for name in require_unchanged:
            if not any(item["path"] == name and item["status"] == "unchanged" for item in comparison):
                errors.append("Unexpected core change from baseline: " + name)
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "file_count": len(actual), "manifest_entries": len(checks),
            "manifest_passed": sum(item["passed"] for item in checks),
            "manifest_coverage_pass": actual == expected, "manifest_checks": checks,
            "deliverables": required, "python_syntax": syntax, "csv": csv_result,
            "baseline_comparison": comparison, "model_executed": False, "container_executed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--rows", type=int, default=9, help="Expected submission.csv rows (default: 9)")
    parser.add_argument("--baseline-manifest", type=Path, help="Prior SHA-256 manifest; does not establish its trustworthiness")
    parser.add_argument("--require-core-unchanged", action="store_true", help="With baseline, require the six model/scorer/CSV/dependency artifacts to match")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    package, output = args.package_root.resolve(), args.output.resolve()
    if not package.is_dir() or args.rows < 1:
        parser.error("package_root must be a directory and --rows must be positive")
    if package == output or package in output.parents:
        parser.error("JSON output must be outside the read-only package")
    if args.baseline_manifest and output == args.baseline_manifest.resolve():
        parser.error("JSON output must not overwrite baseline manifest")
    if args.require_core_unchanged and not args.baseline_manifest:
        parser.error("--require-core-unchanged requires --baseline-manifest")
    try:
        result = audit(package, args.rows, args.baseline_manifest, CORE if args.require_core_unchanged else ())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {"status": "FAIL", "errors": [f"Invalid package: {type(exc).__name__}: {exc}"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": result["errors"]}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
