"""Two fixed 128-mask probes against saved MPS FP32 references; not nine-image scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from explainer import MASK_BATCH_SIZE, make_mask_batch
from runtime_support import MODEL_ROOT, atomic_json, environment, selected_device, sha, verify_sources


def sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps": torch.mps.synchronize()


def comparison(expected, actual, target):
    if not isinstance(target, (int, np.integer)) or not 0 <= int(target) < 5:
        raise ValueError("Target must be one integer supplied class among five")
    for values in (expected, actual):
        for key, shape in (("logits", (1, 5)), ("masked_logits", (128, 5))):
            array = np.asarray(values[key])
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError("Comparison requires FP32 original 1x5 and masked 128x5 logits")
    old = expected["logits"].astype(np.float64)
    new = actual["logits"].astype(np.float64)
    old_masked = expected["masked_logits"].astype(np.float64)
    new_masked = actual["masked_logits"].astype(np.float64)
    old_drop = old[0, target] - old_masked[:, target]
    new_drop = new[0, target] - new_masked[:, target]
    denominator = float(np.linalg.norm(old_drop))
    numerator = float(np.linalg.norm(new_drop - old_drop))
    relative_l2 = numerator / denominator if denominator else (0.0 if numerator == 0 else None)
    checks = {
        "finite": bool(np.isfinite(new).all() and np.isfinite(new_masked).all()),
        "original_all5_close": bool(np.allclose(new, old, rtol=1e-3, atol=1e-4)),
        "masked_all5_close": bool(np.allclose(new_masked, old_masked, rtol=1e-3, atol=1e-4)),
        "original_class_same": bool(np.array_equal(old.argmax(1), new.argmax(1))),
        "masked_classes_same": bool(np.array_equal(old_masked.argmax(1), new_masked.argmax(1))),
        "target_drop_relative_l2_le_1pct": relative_l2 is not None and relative_l2 <= .01,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "rtol": 1e-3, "atol": 1e-4, "target": target,
            "original_max_abs_difference": float(np.max(np.abs(old-new))),
            "masked_max_abs_difference": float(np.max(np.abs(old_masked-new_masked))),
            "target_drop_reference_l2": denominator, "target_drop_difference_l2": numerator,
            "target_drop_relative_l2": relative_l2,
            "target_drop_sign_disagreements": int(np.count_nonzero(np.signbit(old_drop) != np.signbit(new_drop)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=MODEL_ROOT / "checkpoints/best_resnet50.pth")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("Choose fresh probe output paths")
    started = time.monotonic()
    device = selected_device(args.device)
    binding = verify_sources(args.weights)
    with np.load(args.reference, allow_pickle=False) as saved:
        reference = {key: saved[key].copy() for key in saved.files}
    indices = np.random.Generator(np.random.PCG64(20261005)).choice(9408, 128, replace=False)
    if not np.array_equal(indices, reference["indices"]):
        raise RuntimeError("Reference indices differ from independent PCG64 seed20261005 sampling")
    for name in ("original", "perturbed"):
        x = reference[name+"_input"]
        if x.dtype != np.float32 or x.shape != (1, 3, 224, 224) or not np.isfinite(x).all():
            raise ValueError("Probe reference requires one finite FP32 3x224x224 input")
        for key, shape in (("logits", (1, 5)), ("masked_logits", (128, 5))):
            values = reference[name+"_"+key]
            if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Expected finite FP32 original 1x5 and masked 128x5 reference logits")
        target = reference[name+"_target"]
        if target.shape != (1,) or target.dtype != np.int64 or not 0 <= int(target[0]) < 5:
            raise ValueError("Reference target must be exactly one supplied int64 class among five")
    from kaggle.model.model import load_model
    model = load_model(str(args.weights), device=torch.device("cpu"))
    model.to(device=device, dtype=torch.float32).eval()
    record = {"status": "RUNNING", "scope": "Original and fixed perturbed inputs, each 128 masks; not full nine-image scoring",
              "device": str(device), "precision": "float32", "mask_batch_size": MASK_BATCH_SIZE,
              "indices_seed": 20261005, "source_binding": binding, "environment": environment(),
              "reference_sha256": sha(args.reference), "cases": {}}
    atomic_json(args.output, record)
    observed = {"indices": indices}
    try:
        for name in ("original", "perturbed"):
            x = reference[name+"_input"]
            target = int(reference[name+"_target"].reshape(-1)[0])
            if not 0 <= target < 5: raise ValueError("Target must be one of the supplied five model classes")
            with torch.inference_mode():
                logits = model(torch.from_numpy(x).to(device=device, dtype=torch.float32))
                if logits.dtype != torch.float32 or logits.device.type != device.type or logits.shape != (1, 5):
                    raise RuntimeError("Original forward changed FP32 precision, device or five-class shape")
                original = logits.cpu().numpy().copy()
                if not np.isfinite(original).all(): raise RuntimeError("Original forward produced nonfinite logits")
                batches = []
                for first in range(0, 128, MASK_BATCH_SIZE):
                    masks = make_mask_batch(x, np.float32(x.min()), indices[first:first+MASK_BATCH_SIZE])
                    logits = model(torch.from_numpy(masks).to(device=device, dtype=torch.float32))
                    if logits.dtype != torch.float32 or logits.device.type != device.type or logits.shape != (len(masks), 5):
                        raise RuntimeError("Masked forward changed FP32 precision, device or five-class shape")
                    values = logits.cpu().numpy().copy()
                    if not np.isfinite(values).all(): raise RuntimeError("Masked forward produced nonfinite logits")
                    batches.append(values)
                    print(json.dumps({"case": name, "masks_done": first+len(masks), "masks_total": 128,
                                      "elapsed_seconds": time.monotonic()-started}), flush=True)
            sync(device)
            actual = {"logits": original, "masked_logits": np.concatenate(batches)}
            expected = {key: reference[name+"_"+key] for key in actual}
            record["cases"][name] = comparison(expected, actual, target)
            for key, value in actual.items(): observed[name+"_"+key] = value
            atomic_json(args.output, record)
        np.savez_compressed(args.output.with_suffix(".npz"), **observed)
        record.update(status="PASS" if all(c["status"] == "PASS" for c in record["cases"].values()) else "FAIL",
                      elapsed_seconds=time.monotonic()-started, actual_logits_sha256=sha(args.output.with_suffix(".npz")))
        atomic_json(args.output, record)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        if record["status"] != "PASS": raise SystemExit(1)
    except Exception as exc:
        record.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output, record)
        raise


if __name__ == "__main__": main()
