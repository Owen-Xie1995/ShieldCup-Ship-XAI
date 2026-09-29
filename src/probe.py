"""Audit Stage 22 occlusion on one official image and one tiny perturbation.

This script does not call the nine-image submission generator or scorer.  It
compares the implementation's raw attribution with independently evaluated
target logits for two tiles and checks the exact bytes of every tile mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

import experiment as exp


ROOT = Path(__file__).resolve().parent
STAGE17_DEFAULT = Path("/Users/xiefujun/Downloads/stage17")
WEIGHTS_DEFAULT = ROOT.parent / "開發環境/checkpoints/best_resnet50.pth"
WEIGHT_BYTES = 282_678_575
WEIGHT_SHA256 = "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473"
GENERATOR_SHA256 = "3f0c8830e590f6ad46934c034a5bf75f2d2b1d89baf2dba6683cdc1d9dc61df1"
METRIC_SHA256 = "5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda"
PERTURBATION_SEED = 314159
PERTURBATION_STD = 0.0005
TEST_TILE_INDICES = (0, exp.TILE_COUNT - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def system_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def maxrss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else 1024 * value


def timed_raw(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    """Time all 2352 single-tile model inputs and sample process peak RSS."""
    before = process_rss_bytes()
    highest = [before or 0]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            rss = process_rss_bytes()
            if rss is not None and rss > highest[0]:
                highest[0] = rss
            stop.wait(0.02)

    sampler = threading.Thread(target=sample, name="probe-rss-sampler", daemon=True)
    sampler.start()
    stats: dict = {}
    started = time.monotonic()
    try:
        raw = exp.explain_raw(
            model,
            x[None, ...],
            np.array([target], dtype=np.int64),
            device="cpu",
            inference_batch_size=batch_size,
            stats=stats,
        )
    finally:
        elapsed = time.monotonic() - started
        stop.set()
        sampler.join(timeout=2)
    after = process_rss_bytes()
    measurement = {
        "wall_seconds": elapsed,
        "inference_batch_size": batch_size,
        "masked_inputs_expected": exp.TILE_COUNT,
        "rss_before_bytes": before,
        "rss_after_bytes": after,
        "rss_peak_sampled_bytes": max(highest[0], after or 0),
        "process_maxrss_bytes": maxrss_bytes(),
        "implementation_stats": stats,
    }
    return np.asarray(raw), measurement


def independent_logit(model: torch.nn.Module, inputs: np.ndarray, target: int) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(inputs)))
    return logits[:, target].detach().cpu().numpy()


def audit_case(
    model: torch.nn.Module,
    x: np.ndarray,
    raw: np.ndarray,
    target: int,
    measurement: dict,
) -> dict:
    if x.shape != (3, 224, 224) or x.dtype != np.float32:
        raise AssertionError(f"Unexpected input {x.shape}, {x.dtype}")
    if raw.shape != (1, 3, 224, 224) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise AssertionError(f"Invalid raw attribution {raw.shape}, {raw.dtype}")
    baseline = np.float32(x.min())
    z = float(independent_logit(model, x[None, ...], target)[0])
    coverage = np.zeros(x.shape, dtype=np.uint8)
    saved_masks: dict[int, np.ndarray] = {}
    changed_counts: list[int] = []
    audit_chunk = int(measurement["inference_batch_size"])
    for start in range(0, exp.TILE_COUNT, audit_chunk):
        indices = list(range(start, min(start + audit_chunk, exp.TILE_COUNT)))
        masked_batch = exp.make_masked_batch(x, baseline, indices)
        if masked_batch.shape != (len(indices), 3, 224, 224) or masked_batch.dtype != np.float32:
            raise AssertionError(f"Invalid masked batch at tile {start}")
        for j, index in enumerate(indices):
            channel, row_slice, col_slice = exp.tile_slices(index)
            selected = np.zeros(x.shape, dtype=bool)
            selected[channel, row_slice, col_slice] = True
            if int(selected.sum()) != exp.TILE_AREA:
                raise AssertionError(f"Tile {index} selected {selected.sum()} elements")
            masked = masked_batch[j]
            if not np.all(masked[selected] == baseline):
                raise AssertionError(f"Tile {index} did not fill every selected element with b")
            if not np.array_equal(masked[~selected].view(np.uint8), x[~selected].view(np.uint8)):
                raise AssertionError(f"Tile {index} changed unselected input bytes")
            coverage += selected
            changed_counts.append(int(np.count_nonzero(x[selected] != masked[selected])))
            if index in TEST_TILE_INDICES:
                saved_masks[index] = masked.copy()
    if not np.all(coverage == 1) or len(changed_counts) != exp.TILE_COUNT:
        raise AssertionError("The 2352 tiles do not partition the complete CHW tensor once")
    tile_records = []
    for index in TEST_TILE_INDICES:
        channel, row_slice, col_slice = exp.tile_slices(index)
        selected = np.zeros(x.shape, dtype=bool)
        selected[channel, row_slice, col_slice] = True
        if int(selected.sum()) != exp.TILE_AREA:
            raise AssertionError(f"Tile {index} selected {selected.sum()} elements")
        masked = saved_masks[index]
        selected_replacement_exact = bool(np.all(masked[selected] == baseline))
        unselected_bytes_unchanged = bool(
            np.array_equal(masked[~selected].view(np.uint8), x[~selected].view(np.uint8))
        )
        if not selected_replacement_exact or not unselected_bytes_unchanged:
            raise AssertionError(f"Tile {index} replacement/byte identity failed")
        z_mask = float(independent_logit(model, masked[None, ...], target)[0])
        expected_r = (z - z_mask) / exp.TILE_AREA
        actual_tile = raw[0, channel, row_slice, col_slice]
        raw_tile_constant = bool(np.all(actual_tile == actual_tile.flat[0]))
        actual_r = float(actual_tile.flat[0])
        if not raw_tile_constant or not np.isclose(actual_r, expected_r, rtol=0, atol=2e-6):
            raise AssertionError(
                f"Tile {index}: raw r {actual_r} versus independent (z-z_mask)/64 {expected_r}"
            )
        tile_records.append(
            {
                "tile_index": index,
                "channel": int(channel),
                "row_start": int(row_slice.start),
                "row_stop": int(row_slice.stop),
                "col_start": int(col_slice.start),
                "col_stop": int(col_slice.stop),
                "selected_count": int(selected.sum()),
                "selected_replacement_exact": selected_replacement_exact,
                "unselected_bytes_unchanged": unselected_bytes_unchanged,
                "selected_originally_at_baseline": int(np.count_nonzero(x[selected] == baseline)),
                "selected_actual_value_changes": int(np.count_nonzero(x[selected] != masked[selected])),
                "original_target_logit": z,
                "masked_target_logit": z_mask,
                "expected_r_per_element": expected_r,
                "actual_raw_r_per_element": actual_r,
                "raw_tile_constant": raw_tile_constant,
                "raw_r_error": actual_r - expected_r,
            }
        )
    processed = exp.postprocess_signed(raw)
    if processed.shape != raw.shape or processed.dtype != np.float32 or not np.isfinite(processed).all():
        raise AssertionError("Final attribution shape, dtype or finite-value check failed")
    return {
        "input_sha256": sha256_array(x),
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "baseline_from_entire_chw": float(baseline),
        "input_global_min": float(x.min()),
        "input_channel_minima": [float(v) for v in x.min(axis=(1, 2))],
        "target_zero_based": target,
        "original_target_logit": z,
        "raw_attribution_shape": list(raw.shape),
        "raw_attribution_dtype": str(raw.dtype),
        "raw_attribution_finite": bool(np.isfinite(raw).all()),
        "raw_attribution_sha256": sha256_array(raw),
        "all_tile_masks_audited": len(changed_counts),
        "all_tiles_partition_complete_chw_once": bool(np.all(coverage == 1)),
        "selected_count_per_tile": exp.TILE_AREA,
        "actual_changed_elements_min": min(changed_counts),
        "actual_changed_elements_max": max(changed_counts),
        "attribution_shape": list(processed.shape),
        "attribution_dtype": str(processed.dtype),
        "attribution_finite": bool(np.isfinite(processed).all()),
        "attribution_nonzero_count": int(np.count_nonzero(processed)),
        "attribution_min": float(processed.min()),
        "attribution_max": float(processed.max()),
        "attribution_sha256": sha256_array(processed),
        "tiles": tile_records,
        "benchmark": measurement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage17-root", type=Path, default=STAGE17_DEFAULT)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if not 8 <= args.batch_size <= 32:
        raise ValueError("The probe batch must be in the prescribed safe range 8–32")
    if not 1 <= args.threads <= 10:
        raise ValueError("The probe torch thread count must be in 1..10")
    stage17 = args.stage17_root.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    official = stage17 / "official"
    generator = official / "vendor/kaggle/submmit/submmit.py"
    metric = official / "script/metrix.py"
    if weights.stat().st_size != WEIGHT_BYTES or sha256_file(weights) != WEIGHT_SHA256:
        raise RuntimeError("Official weight identity mismatch")
    if sha256_file(generator) != GENERATOR_SHA256 or sha256_file(metric) != METRIC_SHA256:
        raise RuntimeError("Official generator/metric identity mismatch")
    sys.path.insert(0, str(official / "vendor"))
    from kaggle.model.dataset.ship_image_datasets import ShipImageDataset
    from kaggle.model.dataset.transforms import val_transforms
    from kaggle.model.model import load_model

    torch.set_num_threads(args.threads)
    model = load_model(str(weights), device=torch.device("cpu"))
    model.eval()
    dataset = ShipImageDataset(
        str(official / "data/test.csv"), str(official / "data/test_images"), transform=val_transforms
    )
    tensor, label = dataset[0]
    original = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    target = int(label)
    if original.shape != (3, 224, 224) or not np.isfinite(original).all():
        raise RuntimeError("The first officially transformed image is invalid")
    image_name = str(dataset.annotations.iloc[0]["image"])
    image_path = official / "data/test_images" / image_name
    rng = np.random.default_rng(PERTURBATION_SEED)
    perturbation = rng.normal(0.0, PERTURBATION_STD, size=original.shape).astype(np.float32)
    perturbed = np.ascontiguousarray(original + perturbation, dtype=np.float32)
    if np.array_equal(original, perturbed) or np.float32(original.min()) == np.float32(perturbed.min()):
        raise RuntimeError("The deterministic perturbation failed to change the current input baseline")

    print("Auditing 2352 original-image masks with official Mac weights", flush=True)
    original_raw, original_measurement = timed_raw(model, original, target, args.batch_size)
    original_record = audit_case(model, original, original_raw, target, original_measurement)
    print(
        f"Original: {original_measurement['wall_seconds']:.3f}s; sampled peak RSS "
        f"{original_measurement['rss_peak_sampled_bytes']} bytes",
        flush=True,
    )
    print("Auditing all 2352 masks for deterministic tiny perturbation", flush=True)
    perturbed_raw, perturbed_measurement = timed_raw(model, perturbed, target, args.batch_size)
    perturbed_record = audit_case(model, perturbed, perturbed_raw, target, perturbed_measurement)
    total_memory = system_memory_bytes()
    peak = max(
        original_measurement["rss_peak_sampled_bytes"],
        perturbed_measurement["rss_peak_sampled_bytes"],
    )
    recommendation = 8 if total_memory is not None and peak > total_memory / 2 else args.batch_size
    record = {
        "status": "PASS",
        "scope": "first supplied official image and one deterministic tiny perturbation; no nine-image scoring",
        "sources": {
            "stage17": str(stage17),
            "weights": {"path": str(weights), "bytes": weights.stat().st_size, "sha256": sha256_file(weights)},
            "official_generator": {"path": str(generator), "sha256": sha256_file(generator)},
            "official_metric": {"path": str(metric), "sha256": sha256_file(metric)},
            "experiment": {"path": str(Path(exp.__file__).resolve()), "sha256": sha256_file(Path(exp.__file__))},
            "probe": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "official_image": {"path": str(image_path), "sha256": sha256_file(image_path)},
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "physical_memory_bytes": total_memory,
        },
        "perturbation": {
            "seed": PERTURBATION_SEED,
            "distribution": "independent normal noise added to transformed float32 CHW",
            "std": PERTURBATION_STD,
            "realized_linf": float(np.max(np.abs(perturbed - original))),
            "noise_sha256": sha256_array(perturbation),
            "baseline_changed": bool(original.min() != perturbed.min()),
        },
        "original": original_record,
        "perturbed": perturbed_record,
        "inference_batch_size_recommended": recommendation,
        "all_checks_passed": True,
    }
    lines = [
        "Stage 22 occlusion implementation probe: PASS",
        f"Official first image: {image_path}",
        f"Target class (zero-based): {target}",
        f"Inference batch {args.batch_size}; PyTorch CPU threads {args.threads}",
        f"Original b = {original.min():.12f}; tiny-perturbation b = {perturbed.min():.12f}",
        f"Original 2352 masks: {original_measurement['wall_seconds']:.3f}s, peak RSS {original_measurement['rss_peak_sampled_bytes']} bytes",
        f"Perturbed 2352 masks: {perturbed_measurement['wall_seconds']:.3f}s, peak RSS {perturbed_measurement['rss_peak_sampled_bytes']} bytes",
        f"Recommended inference batch: {recommendation}",
        "All 2352 tile masks per input: exactly 64 selected positions replaced; unselected bytes unchanged; complete CHW partition. Independent target logits and r verified for first and last tiles in each input.",
        "Raw and final attributions are finite float32 NCHW; full values and source hashes: probe.json",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output_dir / "probe.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    atomic_text(args.output_dir / "probe.log", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
