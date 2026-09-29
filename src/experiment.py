"""Stage 22: single-channel 8x8 black-occlusion attribution on official images.

The unmodified official submission generator and scorer produce the final CSV and
score. Every callback input, including each Quantus perturbation, is attributed
from its own complete bytes or an exact-content, verified disk checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import random
import secrets
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


METHOD = "black_occlusion8_signed"
METHOD_VERSION = "stage22-8x8-black-occlusion-signed-v1"
TILE_SIZE = 8
CHANNELS = 3
IMAGE_SIZE = 224
TILES_PER_AXIS = IMAGE_SIZE // TILE_SIZE
TILE_COUNT = CHANNELS * TILES_PER_AXIS * TILES_PER_AXIS
TILE_AREA = TILE_SIZE * TILE_SIZE
DEFAULT_INFERENCE_BATCH_SIZE = 32
TORCH_THREADS = 8
WEIGHT_BYTES = 282_678_575
WEIGHT_SHA256 = "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473"
GENERATOR_SHA256 = "3f0c8830e590f6ad46934c034a5bf75f2d2b1d89baf2dba6683cdc1d9dc61df1"
METRIX_SHA256 = "5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda"
CACHE_SCHEMA = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_chw(x_chw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_chw)
    if x.dtype != np.float32 or x.shape != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected float32 CHW (3,224,224), got {x.dtype} {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Input contains non-finite values")
    return x


def tile_slices(index: int) -> tuple[int, slice, slice]:
    """Return (channel, row slice, column slice) in channel-major tile order."""
    index = int(index)
    if not 0 <= index < TILE_COUNT:
        raise IndexError(f"Tile index {index} outside 0..{TILE_COUNT - 1}")
    channel, within_channel = divmod(index, TILES_PER_AXIS * TILES_PER_AXIS)
    tile_row, tile_col = divmod(within_channel, TILES_PER_AXIS)
    y = tile_row * TILE_SIZE
    x = tile_col * TILE_SIZE
    return channel, slice(y, y + TILE_SIZE), slice(x, x + TILE_SIZE)


def make_masked_batch(
    x_chw: np.ndarray, baseline: float, tile_indices: list[int] | range
) -> np.ndarray:
    """Copy one CHW image per tile, replacing exactly one selected tile in each.

    Selected entries that already equal the baseline naturally do not change
    numerically; all unselected float32 bytes are copied without alteration.
    """
    x = _checked_chw(x_chw)
    if not np.isfinite(baseline):
        raise ValueError("Baseline must be finite")
    indices = list(tile_indices)
    batch = np.repeat(np.ascontiguousarray(x)[None, ...], len(indices), axis=0)
    for row, index in enumerate(indices):
        channel, ys, xs = tile_slices(index)
        batch[row, channel, ys, xs] = np.float32(baseline)
    return batch


def _target_logits(model: torch.nn.Module, batch: np.ndarray, target: int, device: str) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(batch).to(device))
        if logits.ndim != 2 or not 0 <= target < logits.shape[1] or logits.shape[0] != batch.shape[0]:
            raise RuntimeError("Model returned invalid logits for the requested target")
        selected = logits[:, target].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(selected).all():
        raise RuntimeError("Non-finite target logit")
    return selected


def _increment(stats: dict | None, key: str, value: int | float) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + value


def explain_raw(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: str = "cpu",
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    stats: dict | None = None,
) -> np.ndarray:
    """Return signed logit-drop attribution before sparsification and smoothing.

    Each image receives its own CHW-minimum baseline, original target logit,
    and 2352 independent one-channel, one-tile occlusion evaluations.
    """
    current = np.asarray(inputs)
    labels = np.asarray(targets)
    if (current.dtype != np.float32 or current.ndim != 4
            or current.shape[1:] != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            or labels.size != current.shape[0] or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("Expected float32 NCHW (N,3,224,224) and one integer target per image")
    if current.shape[0] < 1 or not np.isfinite(current).all():
        raise ValueError("Input batch must be nonempty and finite")
    if device != "cpu":
        raise ValueError("Stage 22 experiment requires CPU")
    if not 1 <= inference_batch_size <= 32:
        raise ValueError("inference_batch_size must be in 1..32")
    labels = labels.reshape(-1).astype(np.int64)
    model.eval()
    result = np.empty_like(current, dtype=np.float32)
    began = time.monotonic()
    for sample_index in range(len(current)):
        x = _checked_chw(current[sample_index])
        target = int(labels[sample_index])
        baseline = np.float32(x.min())
        original = _target_logits(model, np.ascontiguousarray(x[None, ...]), target, device)
        z = float(original[0])
        _increment(stats, "original_inputs_evaluated", 1)
        _increment(stats, "model_forward_calls", 1)
        for first in range(0, TILE_COUNT, inference_batch_size):
            indices = list(range(first, min(first + inference_batch_size, TILE_COUNT)))
            masked = make_masked_batch(x, baseline, indices)
            z_mask = _target_logits(model, masked, target, device)
            _increment(stats, "masked_inputs_evaluated", len(indices))
            _increment(stats, "model_forward_calls", 1)
            for row, tile_index in enumerate(indices):
                channel, ys, xs = tile_slices(tile_index)
                result[sample_index, channel, ys, xs] = np.float32((z - z_mask[row]) / TILE_AREA)
        _increment(stats, "computed_attributions", 1)
    _increment(stats, "attribution_elapsed_seconds", time.monotonic() - began)
    if not np.isfinite(result).all():
        raise RuntimeError("Occlusion returned non-finite attribution")
    return result


def _top_fraction_signed(values: np.ndarray, fraction: float) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    kept = np.zeros_like(flat)
    count = int(np.ceil(flat.shape[1] * fraction))
    for i, row in enumerate(flat):
        indices = np.argsort(np.abs(row), kind="stable")[-count:]
        kept[i, indices] = row[indices]
    return kept.reshape(values.shape)


def postprocess_signed(raw: np.ndarray) -> np.ndarray:
    """CHW-wide abs-ranked top 25%, reflect mean5, abs-ranked top 10%."""
    value = np.asarray(raw)
    if (value.dtype != np.float32 or value.ndim != 4
            or value.shape[1:] != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            or not np.isfinite(value).all()):
        raise ValueError("Expected finite float32 NCHW (N,3,224,224) raw attribution")
    first = _top_fraction_signed(value, 0.25)
    tensor = torch.from_numpy(np.ascontiguousarray(first))
    smooth = F.avg_pool2d(F.pad(tensor, (2, 2, 2, 2), mode="reflect"), 5, stride=1).numpy()
    last = np.ascontiguousarray(_top_fraction_signed(smooth, 0.10), dtype=np.float32)
    if last.shape != value.shape or not np.isfinite(last).all():
        raise RuntimeError("Postprocessing returned invalid attribution")
    return last


def explain(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    method: str = METHOD,
    device: str = "cpu",
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    stats: dict | None = None,
    **kwargs,
) -> np.ndarray:
    if method != METHOD:
        raise ValueError(f"Unknown method {method!r}")
    raw = explain_raw(model, inputs, targets, device, inference_batch_size, stats)
    return postprocess_signed(raw)


def explain_pair(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: str = "cpu",
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return both the unprocessed signed map and the submitted final map."""
    raw = explain_raw(model, inputs, targets, device, inference_batch_size, stats)
    return raw, postprocess_signed(raw)


def occlusion_core_sha256() -> str:
    """Hash source for mask construction, logit differences, and postprocessing.

    The cache specification separately includes all relevant parameters. The
    full program hash also binds helper and submission-wrapper changes.
    """
    functions = (_checked_chw, tile_slices, make_masked_batch, _target_logits,
                 explain_raw, _top_fraction_signed, postprocess_signed, explain_pair)
    source = "\n\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes(order="C")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AttributionCache:
    """Verified exact-input raw/final disk checkpoints, written atomically."""

    def __init__(self, root: Path, weight_sha256: str, code_sha256: str, inference_batch_size: int) -> None:
        if torch.get_num_threads() != TORCH_THREADS:
            raise RuntimeError(f"Expected {TORCH_THREADS} PyTorch CPU threads before cache use")
        self.root = root / METHOD
        self.spec = {
            "schema": CACHE_SCHEMA,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "weight_sha256": weight_sha256,
            "code_sha256": code_sha256,
            "occlusion_core_sha256": occlusion_core_sha256(),
            "tile_size": TILE_SIZE,
            "tile_count": TILE_COUNT,
            "tile_area": TILE_AREA,
            "occlusion": "one channel and one nonoverlapping tile per masked input",
            "baseline": "minimum scalar of each current input's complete CHW tensor",
            "logit": "provided target class logit; signed original minus masked divided by tile area",
            "noise": "none",
            "inference_batch_size": inference_batch_size,
            "torch_num_threads": TORCH_THREADS,
            "top_fraction_first": 0.25,
            "top_fraction_last": 0.10,
            "ranking": "absolute value with original signs retained",
            "smooth": "reflect padded per-channel spatial 5x5 arithmetic mean",
            "stored_arrays": ["raw_attribution", "attribution"],
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
        self._memo: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0, "writes": 0, "invalid_entries": 0}
        _atomic_json(self.root / "config.json", self.spec)

    def _key(self, inputs: np.ndarray, targets: np.ndarray) -> str:
        current = np.ascontiguousarray(np.asarray(inputs))
        labels = np.ascontiguousarray(np.asarray(targets))
        identity = {
            "spec": self.spec,
            "input_shape": list(current.shape),
            "input_dtype": current.dtype.str,
            "input_sha256": hashlib.sha256(_array_bytes(current)).hexdigest(),
            "target_shape": list(labels.shape),
            "target_dtype": labels.dtype.str,
            "target_sha256": hashlib.sha256(_array_bytes(labels)).hexdigest(),
        }
        return hashlib.sha256(_json_bytes(identity)).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    @staticmethod
    def _load(path: Path, key: str, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as archive:
            if sorted(archive.files) != ["attribution", "metadata", "raw_attribution"]:
                raise ValueError("Wrong cache fields")
            raw = archive["raw_attribution"]
            value = archive["attribution"]
            meta = json.loads(str(archive["metadata"].item()))
        if (meta.get("key") != key or meta.get("method_version") != METHOD_VERSION
                or tuple(meta.get("shape", [])) != expected_shape
                or meta.get("dtype") != np.dtype("float32").str
                or raw.shape != expected_shape or raw.dtype != np.float32
                or not np.isfinite(raw).all()
                or value.shape != expected_shape or value.dtype != np.float32
                or not np.isfinite(value).all()
                or meta.get("raw_attribution_sha256") != hashlib.sha256(_array_bytes(raw)).hexdigest()
                or meta.get("attribution_sha256") != hashlib.sha256(_array_bytes(value)).hexdigest()):
            raise ValueError("Cache identity, shape, dtype, raw/final value, or checksum mismatch")
        reproduced = postprocess_signed(raw)
        if not np.array_equal(reproduced.view(np.uint8), value.view(np.uint8)):
            raise ValueError("Cached final map is not the prescribed postprocessing of the raw map")
        return np.ascontiguousarray(raw), np.ascontiguousarray(value)

    def _save(self, path: Path, key: str, raw: np.ndarray, value: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "key": key,
            "method_version": METHOD_VERSION,
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "raw_attribution_sha256": hashlib.sha256(_array_bytes(raw)).hexdigest(),
            "attribution_sha256": hashlib.sha256(_array_bytes(value)).hexdigest(),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, raw_attribution=raw, attribution=value,
                                    metadata=np.array(json.dumps(meta, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            self._load(temporary, key, tuple(value.shape))
            os.replace(temporary, path)
            self._load(path, key, tuple(value.shape))
        finally:
            temporary.unlink(missing_ok=True)

    def get_both_or_compute(self, inputs: np.ndarray, targets: np.ndarray, compute) -> tuple[np.ndarray, np.ndarray]:
        """Return validated raw and final maps, calculating them if absent.

        `compute` must return (raw, final), so a cache miss cannot silently
        store a postprocessed map as a raw attribution.
        """
        key = self._key(inputs, targets)
        saved = self._memo.get(key)
        if saved is not None:
            self.stats["memory_hits"] += 1
            return saved[0].copy(), saved[1].copy()
        path = self._path(key)
        if path.exists():
            try:
                saved = self._load(path, key, tuple(np.asarray(inputs).shape))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, zipfile.BadZipFile):
                self.stats["invalid_entries"] += 1
            else:
                self.stats["disk_hits"] += 1
                self._memo[key] = saved
                return saved[0].copy(), saved[1].copy()
        self.stats["misses"] += 1
        raw_computed, final_computed = compute()
        raw = np.ascontiguousarray(np.asarray(raw_computed))
        result = np.ascontiguousarray(np.asarray(final_computed))
        expected_shape = np.asarray(inputs).shape
        if (raw.shape != expected_shape or raw.dtype != np.float32 or not np.isfinite(raw).all()
                or result.shape != expected_shape or result.dtype != np.float32
                or not np.isfinite(result).all()):
            raise RuntimeError("Computed raw/final attributions have invalid shape, dtype, or finite values")
        # A stored final map must be reproducible from the stored signed raw map.
        reproduced = postprocess_signed(raw)
        if not np.array_equal(reproduced.view(np.uint8), result.view(np.uint8)):
            raise RuntimeError("Final attribution differs from postprocessing the stored raw map")
        self._save(path, key, raw, result)
        self.stats["writes"] += 1
        self._memo[key] = (raw, result)
        return raw.copy(), result.copy()

    def get_or_compute(self, inputs: np.ndarray, targets: np.ndarray, compute) -> np.ndarray:
        """Compatibility API used by the official callback and heat-map tools."""
        _, final = self.get_both_or_compute(inputs, targets, compute)
        return final


def _find_stage17(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if (candidate / "quick_eval.py").is_file() and (candidate / "official/score_c3.py").is_file():
            return candidate
        raise FileNotFoundError(f"Invalid stage17 directory: {candidate}")
    here = Path(__file__).resolve()
    candidates = [parent / "stage17" for parent in here.parents]
    candidates += [parent for parent in here.parents if parent.name == "stage17"]
    for candidate in candidates:
        if (candidate / "quick_eval.py").is_file() and (candidate / "official/score_c3.py").is_file():
            return candidate
    raise FileNotFoundError("stage17 was not found; pass --stage17-root")


def _find_weight(stage17: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = [stage17 / "official/checkpoints/best_resnet50.pth"]
    candidates.extend(stage17.parent.glob("*/開發環境/checkpoints/best_resnet50.pth"))
    candidates.extend(stage17.parent.glob("*/checkpoints/best_resnet50.pth"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("best_resnet50.pth was not found; pass --weights")


def _versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "interpreter": sys.executable}
    for name in ("torch", "torchvision", "captum", "quantus", "numpy", "pandas"):
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:
            versions[name] = f"unavailable: {type(error).__name__}: {error}"
    return versions


def run(
    stage17_root: Path,
    weights: Path,
    output_dir: Path,
    eval_seed: int = 42,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    cache_dir: Path | None = None,
    replace_existing: bool = False,
) -> dict:
    """Run official generator/scorer and finalize outputs only after full success."""
    if not 1 <= inference_batch_size <= 32:
        raise ValueError("inference_batch_size must be in 1..32")
    stage17_root = stage17_root.expanduser().resolve()
    weights = weights.expanduser().resolve()
    if weights.stat().st_size != WEIGHT_BYTES or sha256(weights) != WEIGHT_SHA256:
        raise ValueError("Official checkpoint size or SHA-256 mismatch")
    official = stage17_root / "official"
    generator_path = official / "vendor/kaggle/submmit/submmit.py"
    metrix_path = official / "script/metrix.py"
    if sha256(generator_path) != GENERATOR_SHA256:
        raise ValueError(f"Official generator SHA-256 mismatch: {generator_path}")
    if sha256(metrix_path) != METRIX_SHA256:
        raise ValueError(f"Official metrix.py SHA-256 mismatch: {metrix_path}")
    sys.path.insert(0, str(official / "vendor"))
    from kaggle.model.dataset.ship_image_datasets import ShipImageDataset
    from kaggle.model.dataset.transforms import val_transforms
    from kaggle.model.model import load_model
    from kaggle.submmit.submmit import generate_xai_submission
    from torch.utils.data import DataLoader
    import pandas as pd

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    torch.set_num_threads(TORCH_THREADS)
    model = load_model(str(weights), device=torch.device("cpu"))
    model.eval()
    dataset = ShipImageDataset(
        str(official / "data/test.csv"), str(official / "data/test_images"),
        transform=val_transforms,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    if len(loader) != 9:
        raise RuntimeError(f"Expected exactly nine official images; found {len(loader)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = METHOD if eval_seed == 42 else f"{METHOD}_seed{eval_seed}"
    csv = output_dir / f"{stem}.csv"
    pending = output_dir / f"{stem}.pending.csv"
    score_json = output_dir / f"{stem}.score.json"
    score_pending = output_dir / f"{stem}.score.pending.json"
    run_json = output_dir / f"{stem}.run.json"
    if not replace_existing and any(path.exists() for path in (csv, score_json, run_json)):
        raise FileExistsError(f"Completed output exists for {stem}; pass --replace-existing to rerun")
    pending.unlink(missing_ok=True)
    score_pending.unlink(missing_ok=True)
    started = time.monotonic()
    implementation_hash = sha256(Path(__file__))
    cache = AttributionCache(
        cache_dir or output_dir.parent / "cache", WEIGHT_SHA256,
        implementation_hash, inference_batch_size,
    )
    occlusion_stats: dict = {}

    def callback(model, inputs, targets, device="cpu", **kwargs):
        if device != "cpu":
            raise RuntimeError(f"Stage 22 requires CPU, got {device!r}")
        # The exact callback input and target bytes define the checkpoint.
        current = np.ascontiguousarray(np.asarray(inputs))
        labels = np.ascontiguousarray(np.asarray(targets))
        return cache.get_or_compute(
            current, labels,
            lambda: explain_pair(model, current, labels, device,
                                 inference_batch_size, occlusion_stats),
        )

    try:
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            torch.backends.mps, "is_available", return_value=False
        ):
            generate_xai_submission(model, loader, callback, output_path=str(pending))
        frame = pd.read_csv(pending)
        expected = ["Id", "Faithfulness", "Max_Sensitivity", "Sparsity"]
        if (frame.columns.tolist() != expected or frame.Id.tolist() != list(range(1, 10))
                or not np.isfinite(frame[expected[1:]].to_numpy(dtype=float)).all()):
            raise RuntimeError("Official generator returned invalid nine-image CSV")
        subprocess.run(
            [sys.executable, str(official / "score_c3.py"), str(pending), str(score_pending)],
            check=True,
        )
        scored = json.loads(score_pending.read_text(encoding="utf-8"))
        if scored["sample_count"] != 9 or scored["submission_sha256"] != sha256(pending):
            raise RuntimeError("Official scorer did not verify the generated CSV")
        independent_score = float(
            20.0 * (1.0 + frame["Faithfulness"].to_numpy(dtype=float).mean())
            + 30.0 * np.maximum(1.0 - frame["Max_Sensitivity"].to_numpy(dtype=float) / 2.0, 0.0).mean()
            + 30.0 * frame["Sparsity"].to_numpy(dtype=float).mean()
        )
        if not np.isclose(independent_score, scored["official_score_0_to_100"], rtol=0, atol=1e-9):
            raise RuntimeError("Independent formula disagrees with official scorer")
        pending.replace(csv)
        score_pending.replace(score_json)
    except Exception:
        pending.unlink(missing_ok=True)
        score_pending.unlink(missing_ok=True)
        raise
    means = {name: float(frame[name].mean()) for name in expected[1:]}
    contributions = {
        "faithfulness": 20.0 * (1.0 + means["Faithfulness"]),
        "sensitivity": float(30.0 * np.maximum(1.0 - frame["Max_Sensitivity"].to_numpy() / 2.0, 0.0).mean()),
        "sparsity": 30.0 * means["Sparsity"],
    }
    try:
        import resource
        peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        peak_rss_bytes = None
    record = {
        "status": "OFFICIAL_GENERATOR_AND_FORMULA_NINE_IMAGE_RESULT",
        "scope": "nine supplied images only; unseen platform result unknown",
        "method": METHOD,
        "method_code": METHOD_VERSION,
        "attribution_is_dynamic": True,
        "official_weight_sha256": WEIGHT_SHA256,
        "submission_csv_sha256": sha256(csv),
        "official_formula_score": scored["official_score_0_to_100"],
        "metric_means": means,
        "score_contributions": contributions,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "evaluation_seed": eval_seed,
        "method_noise": "none",
        "attribution_cache": {
            "root": str(cache.root),
            "key": "SHA-256 of full callback input and target bytes/shape/dtype, checked weight and full program/core hashes, tile/postprocess/batch settings, torch and numpy versions",
            "arrays": "Each completed exact-input checkpoint contains both the unprocessed signed map and final attribution, each SHA-256 checked; atomic write followed by reload verification",
            "spec": cache.spec,
            **cache.stats,
        },
        "occlusion": {
            "tile_size": TILE_SIZE,
            "tile_count_per_input": TILE_COUNT,
            "inference_batch_size": inference_batch_size,
            "torch_num_threads": TORCH_THREADS,
            "torch_num_threads_actual": torch.get_num_threads(),
            "computed_attributions": occlusion_stats.get("computed_attributions", 0),
            "masked_inputs_evaluated": occlusion_stats.get("masked_inputs_evaluated", 0),
            "original_inputs_evaluated": occlusion_stats.get("original_inputs_evaluated", 0),
            "model_forward_calls": occlusion_stats.get("model_forward_calls", 0),
            "attribution_elapsed_seconds": round(occlusion_stats.get("attribution_elapsed_seconds", 0.0), 3),
            "peak_rss_bytes": peak_rss_bytes,
        },
        "batch_size": 1,
        "shuffle": False,
        "device": "cpu",
        "runtime": _versions(),
        "paths": {
            "stage17": str(stage17_root),
            "official_generator": str(generator_path),
            "official_scorer": str(metrix_path),
            "weights": str(weights),
        },
        "code_sha256": {
            "experiment": implementation_hash,
            "official_generator": sha256(generator_path),
            "official_scorer": sha256(metrix_path),
            "official_wrapper": sha256(official / "score_c3.py"),
        },
    }
    _atomic_json(run_json, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage17-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--inference-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    stage17 = _find_stage17(args.stage17_root)
    weights = _find_weight(stage17, args.weights)
    result = run(stage17, weights, args.output_dir, eval_seed=args.eval_seed,
                 inference_batch_size=args.inference_batch_size,
                 cache_dir=args.cache_dir, replace_existing=args.replace_existing)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
