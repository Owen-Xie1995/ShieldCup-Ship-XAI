"""Frozen Stage 23 RISE-inspired signed covariance attribution.

The mask bank is sampled once with a dedicated PCG64 generator. This is an
independent project variant, not the original RISE estimator. The official
generator must call ``explain_pair`` on each current input (including Quantus
perturbations); a verified cache hit is the only permitted reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


METHOD = "rise_cov1024_fixed14_post22"
METHOD_VERSION = "stage23-rise-cov1024-fixed14-post22-v1"
MASK_COUNT = 1024
CHANNELS = 3
GRID_SIZE = 14
IMAGE_SIZE = 224
KEEP_PROBABILITY = 0.5
MASK_SEED = 20260930
PRECISION = "fp32"
POSTPROCESS_VERSION = "stage22-abs25-reflect5-abs10-signed"
BANK_SCHEMA = 1
CACHE_SCHEMA = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checked_input(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x)
    if value.dtype != np.float32 or value.shape != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected float32 CHW (3,224,224), got {value.dtype} {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Input contains non-finite values")
    return np.ascontiguousarray(value)


def postprocess_signed(raw: np.ndarray) -> np.ndarray:
    """Stage 22's exact CHW abs25, per-channel reflect mean5, abs10 map."""
    value = np.asarray(raw)
    if (value.dtype != np.float32 or value.ndim != 4
            or value.shape[1:] != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            or not np.isfinite(value).all()):
        raise ValueError("Expected finite float32 NCHW raw attribution")

    def top_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
        flat = values.reshape(len(values), -1)
        kept = np.zeros_like(flat)
        count = int(np.ceil(flat.shape[1] * fraction))
        for i, row in enumerate(flat):
            indices = np.argsort(np.abs(row), kind="stable")[-count:]
            kept[i, indices] = row[indices]
        return kept.reshape(values.shape)

    first = top_fraction(value, 0.25)
    tensor = torch.from_numpy(np.ascontiguousarray(first))
    smooth = F.avg_pool2d(F.pad(tensor, (2, 2, 2, 2), mode="reflect"), 5, stride=1).numpy()
    result = np.ascontiguousarray(top_fraction(smooth, 0.10), dtype=np.float32)
    if result.shape != value.shape or not np.isfinite(result).all():
        raise RuntimeError("Postprocessing returned invalid attribution")
    return result


def covariance_from_sums(
    sum_occluded_times_drop: np.ndarray,
    sum_drop: float,
    q: np.ndarray,
    n_masks: int,
) -> np.ndarray:
    """Implement mean[(o-q)D]/max(q(1-q),1e-6) using global q and sums."""
    if n_masks < 1:
        raise ValueError("n_masks must be positive")
    value = np.asarray(sum_occluded_times_drop, dtype=np.float64)
    rate = np.asarray(q, dtype=np.float64)
    if value.shape != rate.shape or not np.isfinite(value).all() or not np.isfinite(rate).all():
        raise ValueError("Invalid covariance inputs")
    if not np.isfinite(sum_drop) or np.any(rate < 0) or np.any(rate > 1):
        raise ValueError("Invalid logit sum or q")
    numerator = (value - rate * float(sum_drop)) / n_masks
    denominator = np.maximum(rate * (1.0 - rate), 1e-6)
    return numerator / denominator


class MaskBank:
    """Fixed uint8 Bernoulli grid plus CPU float32 bilinear mask stream."""

    def __init__(self, root: Path, *, generation_batch_size: int = 32) -> None:
        self.root = Path(root)
        self.grid_path = self.root / "grids.uint8.npy"
        self.q_path = self.root / "q.float64.npy"
        self.manifest_path = self.root / "bank.json"
        if generation_batch_size not in (32, 64):
            raise ValueError("generation_batch_size must be 32 or 64")
        self.generation_batch_size = generation_batch_size
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.grid_path.exists():
            rng = np.random.Generator(np.random.PCG64(MASK_SEED))
            grid = np.asarray(rng.random((MASK_COUNT, CHANNELS, GRID_SIZE, GRID_SIZE))
                              < KEEP_PROBABILITY, dtype=np.uint8)
            _atomic_npy(self.grid_path, grid)
        self.grids = np.load(self.grid_path, allow_pickle=False, mmap_mode="r")
        if self.grids.dtype != np.uint8 or self.grids.shape != (MASK_COUNT, CHANNELS, GRID_SIZE, GRID_SIZE):
            raise ValueError("Mask grid has wrong shape or dtype")
        expected = np.asarray(
            np.random.Generator(np.random.PCG64(MASK_SEED)).random(self.grids.shape)
            < KEEP_PROBABILITY, dtype=np.uint8)
        if not np.array_equal(self.grids, expected):
            raise ValueError("Mask grid differs from the fixed PCG64 sample")
        self.grid_array_sha256 = array_sha256(self.grids)
        self.grid_file_sha256 = sha256_file(self.grid_path)
        stream = hashlib.sha256()
        occlusion_sum = np.zeros((CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
        for start in range(0, MASK_COUNT, generation_batch_size):
            mask = self.mask_batch(start, min(start + generation_batch_size, MASK_COUNT))
            stream.update(mask.tobytes(order="C"))
            # Preserve mask order in the float64 sum, independent of chunking.
            for one_mask in mask:
                occlusion_sum += 1.0 - one_mask.astype(np.float64)
        computed_q = occlusion_sum / MASK_COUNT
        if not np.isfinite(computed_q).all() or np.any(computed_q < 0) or np.any(computed_q > 1):
            raise RuntimeError("Invalid mask occlusion probability")
        if not self.q_path.exists():
            _atomic_npy(self.q_path, computed_q)
        self.q = np.load(self.q_path, allow_pickle=False)
        if self.q.dtype != np.float64 or self.q.shape != computed_q.shape or not np.array_equal(self.q, computed_q):
            raise ValueError("Saved q differs from the fixed bank's global mean occlusion")
        self.stream_sha256 = stream.hexdigest()
        self.q_array_sha256 = array_sha256(self.q)
        self.q_file_sha256 = sha256_file(self.q_path)
        manifest = {
            "schema": BANK_SCHEMA,
            "method": METHOD,
            "seed": MASK_SEED,
            "bit_generator": "PCG64",
            "sampling": "np.random.Generator(PCG64(20260930)).random((1024,3,14,14)) < 0.5",
            "grid_shape": list(self.grids.shape),
            "grid_dtype": str(self.grids.dtype),
            "grid_array_sha256": self.grid_array_sha256,
            "grid_file_sha256": self.grid_file_sha256,
            "mask_order": "mask-major, channel-major, row-major, column-major",
            "upsampling": "torch.nn.functional.interpolate on CPU float32; bilinear; size=(224,224); align_corners=False",
            "upsampled_stream_dtype": "float32",
            "upsampled_stream_shape": [MASK_COUNT, CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
            "upsampled_stream_sha256": self.stream_sha256,
            "q_dtype": "float64",
            "q_shape": list(self.q.shape),
            "q_array_sha256": self.q_array_sha256,
            "q_file_sha256": self.q_file_sha256,
            "numpy": np.__version__,
            "torch": torch.__version__,
        }
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if old != manifest:
                raise ValueError("Existing mask bank manifest differs from fixed bank")
        else:
            _atomic_json(self.manifest_path, manifest)
        self.manifest = manifest
        self.identity_sha256 = sha256_bytes(canonical_bytes(manifest))

    def mask_batch(self, start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= MASK_COUNT:
            raise ValueError("Invalid mask range")
        low = np.ascontiguousarray(self.grids[start:stop], dtype=np.float32)
        with torch.no_grad():
            up = F.interpolate(torch.from_numpy(low), size=(IMAGE_SIZE, IMAGE_SIZE),
                               mode="bilinear", align_corners=False)
        mask = np.ascontiguousarray(up.numpy(), dtype=np.float32)
        if mask.shape != (stop - start, CHANNELS, IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(mask).all():
            raise RuntimeError("Invalid upsampled mask")
        return mask


class BudgetExceeded(RuntimeError):
    """An explicit deadline was reached after a safe, saved batch boundary."""


class RiseCache:
    """Exact-content completed attributions, with separate safe partial state."""

    def __init__(self, root: Path, bank: MaskBank, *, weight_sha256: str,
                 device: str, inference_batch_size: int) -> None:
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps or cuda")
        if inference_batch_size not in (32, 64):
            raise ValueError("Only batch size 32 or 64 is pre-registered")
        if len(weight_sha256) != 64:
            raise ValueError("Expected complete weight SHA-256")
        self.bank = bank
        self.root = Path(root) / METHOD
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = {
            "schema": CACHE_SCHEMA,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "weight_sha256": weight_sha256,
            "rise_cov_code_sha256": sha256_file(Path(__file__).resolve()),
            "mask_bank_identity_sha256": bank.identity_sha256,
            "grid_array_sha256": bank.grid_array_sha256,
            "upsampled_stream_sha256": bank.stream_sha256,
            "q_array_sha256": bank.q_array_sha256,
            "mask_count": MASK_COUNT,
            "grid_size": GRID_SIZE,
            "keep_probability": KEEP_PROBABILITY,
            "seed": MASK_SEED,
            "baseline": "minimum scalar of each current complete CHW input",
            "masked_input": "b+(x-b)*M",
            "target_logit_drop": "original target logit minus masked target logit",
            "raw_formula": "mean_j[(1-M_ji-q_i)*D_j]/max(q_i*(1-q_i),1e-6)",
            "postprocess": POSTPROCESS_VERSION,
            "device": device,
            "precision": PRECISION,
            "inference_batch_size": inference_batch_size,
            "numpy": np.__version__,
            "torch": torch.__version__,
        }
        self.spec_sha256 = sha256_bytes(canonical_bytes(self.spec))
        _atomic_json(self.root / f"config_{device}_b{inference_batch_size}.json", self.spec)
        self.stats = {"memory_hits": 0, "disk_hits": 0, "misses": 0,
                      "writes": 0, "invalid_entries": 0, "partial_resumes": 0}
        self._memo: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def identity(self, x: np.ndarray, target: int,
                 callback_inputs: np.ndarray | None = None,
                 callback_targets: np.ndarray | None = None) -> dict:
        x = _checked_input(x)
        inputs = np.asarray(callback_inputs) if callback_inputs is not None else x[None, ...]
        labels = np.asarray(callback_targets) if callback_targets is not None else np.asarray([target], dtype=np.int64)
        if inputs.dtype != np.float32 or inputs.ndim != 4 or inputs.shape[1:] != x.shape:
            raise ValueError("Callback input shape/dtype differs from expected NCHW")
        if not np.issubdtype(labels.dtype, np.integer) or labels.size != inputs.shape[0]:
            raise ValueError("Callback target shape/dtype differs from input batch")
        return {
            "spec_sha256": self.spec_sha256,
            "current_sample_shape": list(x.shape),
            "current_sample_dtype": x.dtype.str,
            "current_sample_sha256": array_sha256(x),
            "callback_input_shape": list(inputs.shape),
            "callback_input_dtype": inputs.dtype.str,
            "callback_input_sha256": array_sha256(inputs),
            "callback_target_shape": list(labels.shape),
            "callback_target_dtype": labels.dtype.str,
            "callback_target_sha256": array_sha256(labels),
            "target_zero_based": int(target),
        }

    def key(self, x: np.ndarray, target: int,
            callback_inputs: np.ndarray | None = None,
            callback_targets: np.ndarray | None = None) -> str:
        identity = self.identity(x, target, callback_inputs, callback_targets)
        return sha256_bytes(canonical_bytes(identity))

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    def partial_path(self, key: str) -> Path:
        return self.path(key).with_suffix(".partial.npz")

    def load(self, key: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
        if key in self._memo:
            self.stats["memory_hits"] += 1
            raw, final = self._memo[key]
            return raw.copy(), final.copy(), {"key": key, "source": "memory"}
        path = self.path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                if sorted(archive.files) != ["attribution", "masked_target_logits", "metadata", "raw_attribution"]:
                    raise ValueError("Wrong completed-cache fields")
                raw = np.ascontiguousarray(archive["raw_attribution"])
                final = np.ascontiguousarray(archive["attribution"])
                logits = np.ascontiguousarray(archive["masked_target_logits"])
                meta = json.loads(str(archive["metadata"].item()))
            if (meta.get("key") != key or meta.get("spec_sha256") != self.spec_sha256
                    or sha256_bytes(canonical_bytes(meta.get("identity"))) != key
                    or raw.shape != (1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE) or raw.dtype != np.float32
                    or final.shape != raw.shape or final.dtype != np.float32
                    or logits.shape != (MASK_COUNT,) or logits.dtype != np.float64
                    or not np.isfinite(raw).all() or not np.isfinite(final).all()
                    or not np.isfinite(logits).all()
                    or meta.get("raw_sha256") != array_sha256(raw)
                    or meta.get("final_sha256") != array_sha256(final)
                    or meta.get("masked_target_logits_sha256") != array_sha256(logits)
                    or not np.isfinite(meta.get("baseline", np.nan))
                    or not np.isfinite(meta.get("original_target_logit", np.nan))):
                raise ValueError("Completed-cache identity, data or hash mismatch")
            reproduced = postprocess_signed(raw)
            if not np.array_equal(reproduced.view(np.uint8), final.view(np.uint8)):
                raise ValueError("Final attribution does not match Stage 22 postprocessing")
        except Exception as exc:
            self.stats["invalid_entries"] += 1
            raise ValueError(f"Invalid completed cache {path}: {exc}") from exc
        self.stats["disk_hits"] += 1
        self._memo[key] = (raw, final)
        return raw.copy(), final.copy(), meta

    def save(self, key: str, x: np.ndarray, target: int, raw: np.ndarray,
             final: np.ndarray, original_target_logit: float,
             masked_target_logits: np.ndarray,
             callback_inputs: np.ndarray | None = None,
             callback_targets: np.ndarray | None = None) -> Path:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.ascontiguousarray(raw, dtype=np.float32)
        final = np.ascontiguousarray(final, dtype=np.float32)
        logits = np.ascontiguousarray(masked_target_logits, dtype=np.float64)
        if raw.shape != (1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE) or final.shape != raw.shape:
            raise ValueError("Wrong raw/final shapes")
        if logits.shape != (MASK_COUNT,) or not np.isfinite(logits).all():
            raise ValueError("Wrong masked-logit shape or values")
        if not np.isfinite(raw).all() or not np.isfinite(final).all():
            raise ValueError("Non-finite attribution")
        if not np.array_equal(postprocess_signed(raw).view(np.uint8), final.view(np.uint8)):
            raise ValueError("Final map differs from fixed postprocessing")
        identity = self.identity(x, target, callback_inputs, callback_targets)
        if sha256_bytes(canonical_bytes(identity)) != key:
            raise ValueError("Save identity differs from exact callback cache key")
        meta = {
            "key": key,
            "spec_sha256": self.spec_sha256,
            "identity": identity,
            "method_version": METHOD_VERSION,
            "input_sha256": array_sha256(_checked_input(x)),
            "target_zero_based": int(target),
            "baseline": float(np.min(x)),
            "original_target_logit": float(original_target_logit),
            "raw_sha256": array_sha256(raw),
            "final_sha256": array_sha256(final),
            "masked_target_logits_sha256": array_sha256(logits),
            "mask_bank_identity_sha256": self.bank.identity_sha256,
            "device": self.spec["device"],
            "precision": PRECISION,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, raw_attribution=raw, attribution=final,
                                    masked_target_logits=logits,
                                    metadata=np.array(json.dumps(meta, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            disk_hits_before = self.stats["disk_hits"]
            self.load(key)
            self.stats["disk_hits"] = disk_hits_before
            # The completed, independently verified checkpoint supersedes the
            # resumable batch state. Never delete a partial on a failed write.
            self.partial_path(key).unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
        self.stats["writes"] += 1
        self._memo[key] = (raw, final)
        return path

    def load_partial(self, key: str) -> dict | None:
        path = self.partial_path(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"].item()))
                sum_od = archive["sum_occluded_times_drop"]
                logits = archive["masked_target_logits"]
            count = int(meta["completed_masks"])
            if (meta.get("key") != key or meta.get("spec_sha256") != self.spec_sha256
                    or not 0 < count < MASK_COUNT or count % self.spec["inference_batch_size"] != 0
                    or sum_od.shape != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE) or sum_od.dtype != np.float64
                    or logits.shape != (count,) or logits.dtype != np.float64
                    or meta.get("sum_od_sha256") != array_sha256(sum_od)
                    or meta.get("logits_sha256") != array_sha256(logits)
                    or not np.isfinite(sum_od).all() or not np.isfinite(logits).all()
                    or not np.isfinite(meta.get("sum_drop", np.nan))
                    or not np.isfinite(meta.get("original_target_logit", np.nan))):
                raise ValueError("Partial cache identity, shape or hash mismatch")
        except Exception as exc:
            raise ValueError(f"Invalid partial cache {path}: {exc}") from exc
        return {"completed_masks": count, "sum_od": sum_od,
                "sum_drop": float(meta["sum_drop"]), "logits": logits,
                "original_target_logit": float(meta["original_target_logit"])}

    def save_partial(self, key: str, completed_masks: int, sum_od: np.ndarray,
                     sum_drop: float, logits: np.ndarray, original_target_logit: float) -> Path:
        if not 0 < completed_masks < MASK_COUNT:
            raise ValueError("Partial count outside 1..1023")
        path = self.partial_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        logits = np.ascontiguousarray(logits[:completed_masks], dtype=np.float64)
        sum_od = np.ascontiguousarray(sum_od, dtype=np.float64)
        meta = {"key": key, "spec_sha256": self.spec_sha256,
                "completed_masks": completed_masks,
                "sum_od_sha256": array_sha256(sum_od),
                "logits_sha256": array_sha256(logits),
                "sum_drop": float(sum_drop),
                "original_target_logit": float(original_target_logit)}
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, sum_occluded_times_drop=sum_od,
                                    masked_target_logits=logits,
                                    metadata=np.array(json.dumps(meta, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self.load_partial(key)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def _target_logits(model: torch.nn.Module, batch: torch.Tensor, target: int) -> np.ndarray:
    with torch.no_grad():
        output = model(batch)
        if output.ndim != 2 or output.shape[0] != batch.shape[0] or not 0 <= target < output.shape[1]:
            raise RuntimeError("Model returned logits with the wrong shape or target")
        if output.device.type != batch.device.type or output.dtype != torch.float32:
            raise RuntimeError("Model output left the requested device or FP32 precision")
        result = output[:, target].detach().to("cpu").numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite target logits")
    return result


def compute_single(
    model: torch.nn.Module,
    x: np.ndarray,
    target: int,
    bank: MaskBank,
    *,
    device: str,
    inference_batch_size: int,
    cache: RiseCache | None = None,
    cache_key: str | None = None,
    progress: Callable[[dict], None] | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, dict]:
    """Evaluate one current input and all 1,024 masked target logits."""
    if device not in ("cpu", "mps", "cuda") or inference_batch_size not in (32, 64):
        raise ValueError("Device or batch size outside fixed experiment profile")
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").lower() in ("1", "true", "yes"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be disabled for a real MPS attribution")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no CPU fallback is permitted")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback is permitted")
    x = _checked_input(x)
    model.eval()
    actual_device = next(model.parameters(), None)
    if actual_device is not None and actual_device.device.type != device:
        raise RuntimeError(f"Model parameters on {actual_device.device}, requested {device}")
    tensor = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)
    b = np.float32(x.min())
    began = time.monotonic()
    partial = cache.load_partial(cache_key) if cache is not None and cache_key is not None else None
    if partial is None:
        z = float(_target_logits(model, tensor, target)[0])
        start = 0
        sum_od = np.zeros((CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
        sum_drop = 0.0
        masked_logits = np.empty(MASK_COUNT, dtype=np.float64)
    else:
        z = partial["original_target_logit"]
        start = partial["completed_masks"]
        sum_od = np.ascontiguousarray(partial["sum_od"])
        sum_drop = partial["sum_drop"]
        masked_logits = np.empty(MASK_COUNT, dtype=np.float64)
        masked_logits[:start] = partial["logits"]
        cache.stats["partial_resumes"] += 1
    for first in range(start, MASK_COUNT, inference_batch_size):
        stop = min(first + inference_batch_size, MASK_COUNT)
        mask_cpu = bank.mask_batch(first, stop)
        mask_device = torch.from_numpy(mask_cpu).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            masked = torch.tensor(b, dtype=torch.float32, device=device) + (tensor - b) * mask_device
            if masked.device.type != device or masked.dtype != torch.float32:
                raise RuntimeError("Masked batch is not FP32 on the requested device")
            values = _target_logits(model, masked, target)
        masked_logits[first:stop] = values
        drops = z - values
        occluded = 1.0 - mask_cpu.astype(np.float64)
        sum_od += np.einsum("n,nchw->chw", drops, occluded, dtype=np.float64, optimize=True)
        sum_drop += float(np.sum(drops, dtype=np.float64))
        del mask_device, masked, occluded
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        if cache is not None and cache_key is not None and stop < MASK_COUNT:
            cache.save_partial(cache_key, stop, sum_od, sum_drop, masked_logits, z)
        if progress is not None:
            progress({"method": METHOD, "device": device, "completed_masks": stop,
                      "planned_masks": MASK_COUNT,
                      "elapsed_seconds": time.monotonic() - began,
                      "last_completed_at_unix": time.time()})
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic and stop < MASK_COUNT:
            raise BudgetExceeded(f"Stopped safely after {stop}/{MASK_COUNT} masks")
    raw_chw = covariance_from_sums(sum_od, sum_drop, bank.q, MASK_COUNT)
    raw = np.ascontiguousarray(raw_chw[None, ...], dtype=np.float32)
    final = postprocess_signed(raw)
    if not np.isfinite(raw).all() or not np.isfinite(final).all():
        raise RuntimeError("Non-finite raw/final attribution")
    detail = {"original_target_logit": z, "baseline": float(b),
              "masked_target_logits_sha256": array_sha256(masked_logits),
              "masked_inputs_evaluated": MASK_COUNT - start,
              "elapsed_seconds": time.monotonic() - began,
              "device": device, "precision": PRECISION,
              "resumed_from_masks": start}
    return raw, final, z, masked_logits, detail


def explain_pair(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    device: str = "cpu",
    inference_batch_size: int = 32,
    bank: MaskBank,
    cache: RiseCache | None = None,
    stats: dict | None = None,
    progress: Callable[[dict], None] | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(inputs)
    labels = np.asarray(targets)
    if (current.dtype != np.float32 or current.ndim != 4
            or current.shape[1:] != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            or not np.isfinite(current).all() or current.shape[0] < 1
            or labels.size != current.shape[0]
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("Expected finite float32 NCHW and one integer target per input")
    if cache is not None and (cache.spec["device"] != device
                              or cache.spec["inference_batch_size"] != inference_batch_size
                              or cache.bank.identity_sha256 != bank.identity_sha256):
        raise ValueError("Cache device, batch size or bank differs from current calculation")
    raw_out: list[np.ndarray] = []
    final_out: list[np.ndarray] = []
    for x, target in zip(current, labels.reshape(-1)):
        target = int(target)
        key = cache.key(x, target, current, labels) if cache is not None else None
        hit = cache.load(key) if cache is not None else None
        if hit is not None:
            raw, final, _ = hit
            if stats is not None:
                stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        else:
            if cache is not None:
                cache.stats["misses"] += 1
            raw, final, z, logits, detail = compute_single(
                model, x, target, bank, device=device,
                inference_batch_size=inference_batch_size,
                cache=cache, cache_key=key, progress=progress,
                deadline_monotonic=deadline_monotonic)
            if cache is not None:
                cache.save(key, x, target, raw, final, z, logits,
                           callback_inputs=current, callback_targets=labels)
            if stats is not None:
                stats["computed_attributions"] = stats.get("computed_attributions", 0) + 1
                stats["masked_inputs_evaluated"] = stats.get("masked_inputs_evaluated", 0) + detail["masked_inputs_evaluated"]
                stats["attribution_elapsed_seconds"] = stats.get("attribution_elapsed_seconds", 0.0) + detail["elapsed_seconds"]
        raw_out.append(raw)
        final_out.append(final)
    return np.concatenate(raw_out, axis=0), np.concatenate(final_out, axis=0)


def explain(model: torch.nn.Module, inputs: np.ndarray, targets: np.ndarray,
            *, method: str = METHOD, **kwargs) -> np.ndarray:
    if method != METHOD:
        raise ValueError(f"Unknown method {method!r}")
    return explain_pair(model, inputs, targets, **kwargs)[1]
