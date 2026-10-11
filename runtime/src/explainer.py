"""Independent FP32 black_occlusion4_soft15 explanation for supplied targets.

This module needs only NumPy, SciPy and PyTorch. It reads no files, maintains
no cache, and uses no random numbers. Masks always use a batch size of 64.
Optional ``progress`` and ``trace_hook`` callables receive dictionaries; they
must not mutate global RNG state if that property is required by the caller.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

METHOD = "black_occlusion4_soft15"
TILE_SIZE = 4
MASK_BATCH_SIZE = 64


def _numpy_copy(value: Any, *, floating: bool = False) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if floating:
            value = value.to(device="cpu", dtype=torch.float32)
        else:
            value = value.cpu()
        value = value.numpy()
    array = np.asarray(value)
    if floating:
        if array.dtype.kind not in "fiu":
            raise ValueError("Inputs must contain real numeric values")
        array = np.asarray(array, dtype=np.float32)
    return np.array(array, copy=True, order="C")


def _shape_check(shape: tuple[int, ...]) -> None:
    if len(shape) != 4 or any(size <= 0 for size in shape):
        raise ValueError("Inputs must be nonempty NCHW")
    if shape[2] % TILE_SIZE or shape[3] % TILE_SIZE:
        raise ValueError("Height and width must be multiples of the fixed 4x4 tile")


def _requested_device(device: Any) -> torch.device:
    requested = torch.device(device)
    if requested.type not in ("cpu", "mps", "cuda"):
        raise ValueError("Supported explicit devices are cpu, mps and cuda")
    if requested.type == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "", "0"):
            raise RuntimeError("MPS CPU fallback must be disabled before starting Python")
        if not torch.backends.mps.is_available():
            raise RuntimeError("The requested MPS device is unavailable")
        if requested.index not in (None, 0):
            raise ValueError("MPS supports only device index 0")
        requested = torch.device("mps", 0)
    elif requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("The requested CUDA device is unavailable")
        index = requested.index
        if index is None:
            index = torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError("The requested CUDA device index is unavailable")
        requested = torch.device("cuda", index)
        if hasattr(torch.backends, "fp32_precision"):
            policies = (
                torch.backends.fp32_precision,
                torch.backends.cuda.matmul.fp32_precision,
                torch.backends.cudnn.fp32_precision,
                torch.backends.cudnn.conv.fp32_precision,
            )
            if any(policy == "tf32" for policy in policies):
                raise RuntimeError("Set CUDA FP32 precision policies to ieee before explaining")
        elif torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("Disable CUDA matmul and cuDNN TF32 before explaining")
    else:
        requested = torch.device("cpu")
    if torch.is_autocast_enabled(requested.type):
        raise RuntimeError("Call the FP32 explainer outside autocast")
    return requested


def make_mask_batch(
    inputs: np.ndarray, baseline: np.float32, indices: Any
) -> np.ndarray:
    """Make independent one-channel masks in fixed channel, y, x order.

    ``inputs`` is one canonical FP32 sample with shape (1,C,H,W). The helper
    also supports the explicitly sampled mask indices used in deployment probes.
    """
    inputs = np.asarray(inputs)
    _shape_check(inputs.shape)
    if len(inputs) != 1 or inputs.dtype != np.float32:
        raise ValueError("Mask construction requires one FP32 NCHW sample")
    ids = list(indices)
    channels, height, width = inputs.shape[1:]
    rows, columns = height // TILE_SIZE, width // TILE_SIZE
    total = channels * rows * columns
    masked = np.repeat(inputs, len(ids), axis=0)
    for batch_row, value in enumerate(ids):
        index = int(value)
        if index != value or not 0 <= index < total:
            raise ValueError("Mask index is outside the current sample")
        channel, within = divmod(index, rows * columns)
        row, column = divmod(within, columns)
        y, x = row * TILE_SIZE, column * TILE_SIZE
        masked[batch_row, channel, y : y + TILE_SIZE, x : x + TILE_SIZE] = baseline
    return masked


def reconstruct_raw(
    logits: np.ndarray,
    masked_logits: np.ndarray,
    target: int,
    sample_shape: tuple[int, ...],
) -> np.ndarray:
    """Allocate signed float64 target-logit drops / 16 as FP32 tile values."""
    _shape_check(sample_shape)
    if sample_shape[0] != 1:
        raise ValueError("Raw reconstruction requires one NCHW sample")
    channels, height, width = sample_shape[1:]
    rows, columns = height // TILE_SIZE, width // TILE_SIZE
    total = channels * rows * columns
    logits = np.asarray(logits)
    masked_logits = np.asarray(masked_logits)
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("Original logits must have shape (1,K)")
    if masked_logits.shape != (total, logits.shape[1]):
        raise ValueError("Masked logits do not match the full fixed 4x4 grid")
    if not 0 <= int(target) < logits.shape[1]:
        raise ValueError("Supplied target is outside the model output classes")
    if not np.isfinite(logits).all() or not np.isfinite(masked_logits).all():
        raise ValueError("Logits must be finite")
    values = (
        (np.float64(logits[0, int(target)]) - masked_logits[:, int(target)].astype(np.float64))
        / 16
    ).astype(np.float32)
    raw = np.repeat(
        np.repeat(values.reshape(channels, rows, columns), TILE_SIZE, axis=1),
        TILE_SIZE,
        axis=2,
    )
    return np.ascontiguousarray(raw[None], dtype=np.float32)


def soft15(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Frozen signed soft15 processing, independently for each NCHW sample.

    ``s``, ``gate`` and ``tau`` use float64. ``final`` is finite FP32 and keeps
    the input shape. This function performs no inference and can be audited
    directly against saved raw arrays.
    """
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    _shape_check(raw.shape)
    if not np.isfinite(raw).all():
        raise ValueError("Raw explanations must be finite")
    s_all = np.empty(raw.shape, dtype=np.float64)
    gate_all = np.empty(raw.shape, dtype=np.float64)
    tau_all = np.empty((len(raw),), dtype=np.float64)
    final = np.empty(raw.shape, dtype=np.float32)
    for index, sample in enumerate(raw):
        s = gaussian_filter(
            np.abs(sample).astype(np.float64),
            sigma=(0, 1, 1),
            mode="reflect",
            truncate=4.0,
        )
        tau = float(np.quantile(s, 0.85, method="linear"))
        gate = np.maximum(1.0 - tau / np.maximum(s, 1e-12), 0.0)
        s_all[index] = s
        tau_all[index] = tau
        gate_all[index] = gate
        final[index] = np.ascontiguousarray(sample.astype(np.float64) * gate, dtype=np.float32)
    if not np.isfinite(final).all():
        raise RuntimeError("The final FP32 explanation is nonfinite")
    return {"s": s_all, "tau": tau_all, "gate": gate_all, "final": final}


def _forward(model: torch.nn.Module, batch: np.ndarray, device: torch.device) -> np.ndarray:
    output = model(torch.from_numpy(batch).to(device=device, dtype=torch.float32))
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("The supplied model must return a logit tensor")
    if output.dtype != torch.float32 or output.device != device:
        raise RuntimeError("Model output must use the requested device and FP32")
    if output.ndim != 2 or output.shape[0] != len(batch) or output.shape[1] <= 0:
        raise RuntimeError("Model logits must have shape (batch,K)")
    values = np.ascontiguousarray(output.detach().cpu().numpy(), dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("The model produced nonfinite logits")
    return values


def custom_explain_func(model, inputs, targets, device, **kwargs) -> np.ndarray:
    """Explain current inputs and supplied targets with fixed FP32 4x4 masks.

    Inputs may be NumPy arrays or tensors, always with nonempty NCHW shape.
    Height and width must be positive multiples of 4. Targets contain exactly
    one integer class per sample. Caller inputs and targets are never modified.
    The supplied model is explicitly moved to ``device`` and FP32, then put in
    eval mode. There is no device fallback or batch-size adjustment.

    Optional keyword arguments:
      progress: callable(dict), before and after each sample and at most every
        25 seconds between mask batches by default. Forward calls are atomic.
      progress_interval_seconds: positive interval no greater than 30 seconds.
      trace_hook: callable(dict), once per completed sample, with isolated
        copies of input, target, original logits, every masked logit, raw,
        final, s, gate and tau, plus device, baseline and mask metadata.
    Other keywords supplied by the official explanation caller are ignored.
    """
    x_all = _numpy_copy(inputs, floating=True)
    _shape_check(x_all.shape)
    if not np.isfinite(x_all).all():
        raise ValueError("Inputs must be finite after FP32 conversion")
    target_values = _numpy_copy(targets).reshape(-1)
    if target_values.size != len(x_all) or target_values.dtype.kind not in "fiu":
        raise ValueError("Supply one integer target for each NCHW sample")
    if not np.isfinite(target_values).all() or not np.equal(target_values, np.floor(target_values)).all():
        raise ValueError("Targets must be finite integer class indices")
    if (target_values < 0).any() or (target_values > np.iinfo(np.int64).max).any():
        raise ValueError("Target class indices are outside the supported range")
    target_values = target_values.astype(np.int64, copy=False)
    progress = kwargs.get("progress")
    trace_hook = kwargs.get("trace_hook")
    for name, callback in (("progress", progress), ("trace_hook", trace_hook)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable")
    interval = float(kwargs.get("progress_interval_seconds", 25.0))
    if not np.isfinite(interval) or not 0 < interval <= 30:
        raise ValueError("Progress interval must be positive and at most 30 seconds")
    requested = _requested_device(device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The supplied model must be a torch.nn.Module")
    model = model.to(device=requested, dtype=torch.float32)
    model.eval()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device != requested:
            raise RuntimeError("Model tensors must use the explicitly requested device")
        if (tensor.is_floating_point() or tensor.is_complex()) and tensor.dtype != torch.float32:
            raise RuntimeError("Floating model tensors must use FP32")
    result = np.empty(x_all.shape, dtype=np.float32)
    started = time.monotonic()
    last_progress = started
    channels, height, width = x_all.shape[1:]
    total = channels * (height // TILE_SIZE) * (width // TILE_SIZE)

    with torch.no_grad():
        for sample_index in range(len(x_all)):
            x = np.ascontiguousarray(x_all[sample_index : sample_index + 1])
            target = int(target_values[sample_index])
            baseline = np.float32(x.min())
            sample_started = time.monotonic()

            def report(done: int, status: str, *, force: bool = False) -> None:
                nonlocal last_progress
                now = time.monotonic()
                if progress is not None and (force or now - last_progress >= interval):
                    sample_elapsed = now - sample_started
                    progress({
                        "method": METHOD,
                        "status": status,
                        "sample_index": sample_index,
                        "samples_total": len(x_all),
                        "masks_done": done,
                        "masks_total": total,
                        "elapsed_seconds": now - started,
                        "sample_elapsed_seconds": sample_elapsed,
                        "eta_seconds": (total - done) * sample_elapsed / done if done else None,
                        "device": str(requested),
                        "dtype": "float32",
                        "mask_batch_size": MASK_BATCH_SIZE,
                    })
                    last_progress = now

            report(0, "original", force=True)
            logits = _forward(model, x, requested)
            if target >= logits.shape[1]:
                raise ValueError("Supplied target is outside the model output classes")
            masked_logits = np.empty((total, logits.shape[1]), dtype=np.float32)
            for first in range(0, total, MASK_BATCH_SIZE):
                end = min(first + MASK_BATCH_SIZE, total)
                masks = make_mask_batch(x, baseline, range(first, end))
                values = _forward(model, masks, requested)
                if values.shape[1] != logits.shape[1]:
                    raise RuntimeError("The model changed its output class count")
                masked_logits[first:end] = values
                report(end, "masking")
            raw = reconstruct_raw(logits, masked_logits, target, x.shape)
            processed = soft15(raw)
            result[sample_index : sample_index + 1] = processed["final"]
            if trace_hook is not None:
                trace_hook({
                    "method": METHOD,
                    "sample_index": sample_index,
                    "device": str(requested),
                    "dtype": "float32",
                    "baseline": float(baseline),
                    "target_index": target,
                    "mask_batch_size": MASK_BATCH_SIZE,
                    "tile_size": TILE_SIZE,
                    "mask_order": "channel,y,x",
                    "input": x.copy(),
                    "target": target_values[sample_index : sample_index + 1].copy(),
                    "logits": logits.copy(),
                    "masked_logits": masked_logits.copy(),
                    "raw": raw.copy(),
                    **{name: value.copy() for name, value in processed.items()},
                })
            report(total, "complete", force=True)
    if result.shape != x_all.shape or result.dtype != np.float32 or not np.isfinite(result).all():
        raise RuntimeError("Explanation must be finite FP32 with the input shape")
    return result
