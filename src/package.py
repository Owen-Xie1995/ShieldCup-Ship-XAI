"""Atomically package Stage 22 deliverables with a local SHA-256 manifest.

MANIFEST_SHA256.json lists all deliverable files except itself, because a file
cannot contain its own hash without recursion. The ZIP contains the identical
manifest. CRC, entry set, sizes, and all listed hashes are independently read
back before the completed ZIP replaces its destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import stat
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.with_suffix(".zip")
METHOD = "black_occlusion8_signed"
MANIFEST_NAME = "MANIFEST_SHA256.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_files() -> list[Path]:
    names = [
        "README.md", "experiment.py", "probe.py", "probe.json", "probe.log",
        "heatmaps.py", "heatmaps/manifest.json", "literature.md",
        "summarize.py", "paired_summary.py", "build_report.py", "verify_outputs.py",
        "record_environment.py", "package.py",
        "per_image_comparison.csv", "summary.json", "branch_decision.json", "report.md",
        "environment.json", "verification.json", f"cache/{METHOD}/config.json",
    ]
    names.extend(f"runs/{METHOD}.{extension}"
                 for extension in ("csv", "score.json", "run.json", "log"))
    return [Path(name) for name in names]


def is_internal(relative: Path) -> bool:
    return any(part in {"__pycache__", ".mplconfig"} for part in relative.parts)


def is_temporary(relative: Path) -> bool:
    name = relative.name
    return name.endswith((".tmp", ".temp", ".partial", ".part", ".pending")) or ".pending." in name


def collect(root: Path) -> list[Path]:
    missing = [str(relative) for relative in required_files()
               if not (root / relative).is_file() or (root / relative).stat().st_size == 0]
    if not any((root / "cache" / METHOD).rglob("*.npz")):
        missing.append(f"cache/{METHOD}/<validated attribution>.npz")
    if missing:
        raise FileNotFoundError("Missing or empty Stage 22 deliverables: " + ", ".join(missing))
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("Independent verification must pass before packaging")
    paths: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if is_internal(relative):
            continue
        if is_temporary(relative):
            raise RuntimeError(f"Unfinished temporary file remains: {path}")
        if path.is_symlink():
            raise ValueError(f"Symlink cannot be packaged safely: {path}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"Nonregular file cannot be packaged: {path}")
        if relative.as_posix() != MANIFEST_NAME:
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def manifest_for(root: Path, paths: list[Path]) -> tuple[dict, bytes]:
    manifest = {
        "schema": 1,
        "archive_root": root.name,
        "hash_algorithm": "SHA-256",
        "self_exclusion": "MANIFEST_SHA256.json cannot include its own hash without recursion; all other archived files are listed.",
        "files": {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        },
    }
    encoded = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return manifest, encoded


def atomic_bytes(path: Path, contents: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_local(root: Path, manifest: dict, manifest_bytes: bytes) -> None:
    if (root / MANIFEST_NAME).read_bytes() != manifest_bytes:
        raise RuntimeError("Local manifest changed after creation")
    actual = {path.relative_to(root).as_posix() for path in collect(root)}
    if actual != set(manifest["files"]):
        raise RuntimeError("Local file set differs from SHA-256 manifest")
    for relative, meta in manifest["files"].items():
        path = root / relative
        if path.stat().st_size != meta["bytes"] or sha256(path) != meta["sha256"]:
            raise RuntimeError(f"Local file changed after manifest: {relative}")


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def verify_zip(path: Path, manifest: dict, manifest_bytes: bytes) -> None:
    prefix = manifest["archive_root"] + "/"
    expected = {prefix + relative for relative in manifest["files"]}
    expected.add(prefix + MANIFEST_NAME)
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise RuntimeError("ZIP entries differ from local SHA-256 manifest")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC check failed: {bad}")
        if archive.read(prefix + MANIFEST_NAME) != manifest_bytes:
            raise RuntimeError("Embedded manifest differs from local manifest")
        for relative, meta in manifest["files"].items():
            digest = hashlib.sha256()
            count = 0
            with archive.open(prefix + relative, "r") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    count += len(chunk)
                    digest.update(chunk)
            if count != meta["bytes"] or digest.hexdigest() != meta["sha256"]:
                raise RuntimeError(f"Archived size or SHA-256 mismatch: {relative}")


def package(root: Path = ROOT, output: Path = DEFAULT_OUTPUT) -> dict:
    root = root.resolve(strict=True)
    if root.name != "output_mac_stage22":
        raise ValueError(f"Expected output_mac_stage22 directory, got {root}")
    output = output.expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("ZIP output must be outside the directory being packaged")
    paths = collect(root)
    manifest, manifest_bytes = manifest_for(root, paths)
    atomic_bytes(root / MANIFEST_NAME, manifest_bytes)
    verify_local(root, manifest, manifest_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as target:
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True, strict_timestamps=True) as archive:
                for path in paths:
                    entry = root.name + "/" + path.relative_to(root).as_posix()
                    with path.open("rb") as source, archive.open(zip_info(entry), "w") as sink:
                        shutil.copyfileobj(source, sink, length=1024 * 1024)
                archive.writestr(zip_info(root.name + "/" + MANIFEST_NAME), manifest_bytes)
            target.flush()
            os.fsync(target.fileno())
        verify_zip(temporary, manifest, manifest_bytes)
        verify_local(root, manifest, manifest_bytes)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "zip": str(output), "zip_bytes": output.stat().st_size,
        "zip_sha256": sha256(output), "file_count": len(paths),
        "manifest_sha256": sha256(root / MANIFEST_NAME),
        "integrity": "Local manifest, ZIP CRC and all archived SHA-256 hashes verified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
