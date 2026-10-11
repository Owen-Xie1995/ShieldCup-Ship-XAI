"""Read-only independent bank, saved-logit and CPU/MPS probe validation.

No model is loaded. The function can be called by the root artifact verifier.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_SHA = "83c064cffdd1cc5993bac43a48dcfa4311d7b89ea1f4bd29f0454ff8aef4b5eb"
PREREG_SHA = "343c7eaaee766c50c868954a4ca7eaf5ba9e9b5eb93d86e4f53198f585fc1d20"
COUNT = 1024


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def postprocess(raw: np.ndarray) -> np.ndarray:
    def top(array: np.ndarray, fraction: float) -> np.ndarray:
        flattened = array.reshape(1, -1)
        selected = np.argsort(np.abs(flattened[0]), kind="stable")[-int(np.ceil(flattened.size * fraction)):]
        kept = np.zeros_like(flattened)
        kept[0, selected] = flattened[0, selected]
        return kept.reshape(array.shape)
    first = top(raw, 0.25)
    averaged = F.avg_pool2d(F.pad(torch.from_numpy(np.ascontiguousarray(first)),
                                 (2, 2, 2, 2), mode="reflect"), 5, stride=1).numpy()
    return np.ascontiguousarray(top(averaged, 0.10), dtype=np.float32)


def verify_candidate_probe(root: Path = ROOT) -> dict:
    root = Path(root)
    acceptance_path = root / "device/probe_acceptance.json"
    require(sha(acceptance_path) == ACCEPTANCE_SHA and sha(root / "preregistered_method.json") == PREREG_SHA,
            "Candidate device limits or preregistration changed")
    acceptance = read_json(acceptance_path)
    limits = acceptance["correctness"]
    with np.load(root / "device/probe_inputs.npz", allow_pickle=False) as archive:
        inputs = {name: np.ascontiguousarray(archive[name]) for name in ("original", "perturbed")}
        target = np.ascontiguousarray(archive["target"])
    require(target.dtype == np.int64 and target.shape == (1,) and target.tolist() == [0],
            "Probe target is not the fixed official first-image target")
    fixed = acceptance["fixed_inputs"]
    require(array_sha(inputs["original"]) == fixed["transformed_float32_chw_sha256"]
            and array_sha(inputs["perturbed"]) == fixed["perturbation"]["perturbed_float32_chw_sha256"]
            and not np.array_equal(inputs["original"], inputs["perturbed"]),
            "Fixed original or perturbed probe input changed")
    bank = root / "candidate/bank"
    grids = np.load(bank / "grids.uint8.npy", allow_pickle=False)
    q = np.load(bank / "q.float64.npy", allow_pickle=False)
    manifest = read_json(bank / "bank.json")
    bank_identity = hashlib.sha256(canonical(manifest)).hexdigest()
    require(grids.shape == (COUNT, 3, 14, 14) and grids.dtype == np.uint8
            and np.array_equal(grids, (np.random.Generator(np.random.PCG64(20260930))
                                      .random(grids.shape) < 0.5).astype(np.uint8)),
            "Fixed PCG64 candidate grids differ")
    require(q.shape == (3, 224, 224) and q.dtype == np.float64 and np.isfinite(q).all()
            and manifest["grid_file_sha256"] == sha(bank / "grids.uint8.npy")
            and manifest["q_file_sha256"] == sha(bank / "q.float64.npy")
            and manifest["grid_array_sha256"] == array_sha(grids)
            and manifest["q_array_sha256"] == array_sha(q), "Mask bank or q content hash differs")
    loaded = []
    for name in ("original", "perturbed"):
        for device in ("cpu", "mps"):
            probe = read_json(root / f"candidate/probes/probe_{device}.json")
            case = probe["cases"][name]
            spec = read_json(root / f"cache/rise_cov1024_fixed14_post22/config_{device}_b32.json")
            spec_digest = hashlib.sha256(canonical(spec)).hexdigest()
            require(probe["status"] == case["status"] == "PASS" and probe["device"] == device
                    and probe["precision"] == "fp32" and probe["batch_size"] == 32
                    and probe["preregistered_method_sha256"] == PREREG_SHA
                    and probe["candidate_code_sha256"] == sha(root / "candidate/rise_cov.py")
                    and probe["probe_code_sha256"] == sha(root / "candidate/probe_candidate.py")
                    and probe["mask_bank_sha256"] == bank_identity
                    and probe["cache_spec_sha256"] == spec_digest
                    and spec["device"] == device and spec["precision"] == "fp32"
                    and spec["mask_bank_identity_sha256"] == bank_identity,
                    f"Probe identity mismatch: {device}/{name}")
            x = inputs[name]
            identity = {"spec_sha256": spec_digest,
                        "current_sample_shape": list(x.shape), "current_sample_dtype": x.dtype.str,
                        "current_sample_sha256": array_sha(x),
                        "callback_input_shape": [1, *x.shape], "callback_input_dtype": x.dtype.str,
                        "callback_input_sha256": array_sha(x[None]),
                        "callback_target_shape": list(target.shape), "callback_target_dtype": target.dtype.str,
                        "callback_target_sha256": array_sha(target), "target_zero_based": 0}
            key = hashlib.sha256(canonical(identity)).hexdigest()
            path = root / "cache/rise_cov1024_fixed14_post22" / key[:2] / f"{key}.npz"
            with np.load(path, allow_pickle=False) as archive:
                require(sorted(archive.files) == ["attribution", "masked_target_logits", "metadata", "raw_attribution"],
                        f"Cache schema mismatch: {device}/{name}")
                raw = np.ascontiguousarray(archive["raw_attribution"])
                final = np.ascontiguousarray(archive["attribution"])
                masked = np.ascontiguousarray(archive["masked_target_logits"])
                meta = json.loads(str(archive["metadata"].item()))
            require(raw.shape == final.shape == (1, 3, 224, 224) and raw.dtype == final.dtype == np.float32
                    and masked.shape == (COUNT,) and masked.dtype == np.float64
                    and np.isfinite(raw).all() and np.isfinite(final).all() and np.isfinite(masked).all(),
                    f"Invalid attribution/logit arrays: {device}/{name}")
            require(meta["key"] == case["cache_key"] == key and meta["identity"] == identity
                    and meta["spec_sha256"] == spec_digest and meta["device"] == case["actual_device"] == device
                    and meta["precision"] == "fp32" and meta["input_sha256"] == case["input_sha256"] == array_sha(x)
                    and meta["target_zero_based"] == case["target_zero_based"] == 0
                    and meta["mask_bank_identity_sha256"] == bank_identity and case["masked_count"] == COUNT
                    and meta["baseline"] == case["baseline"] == float(x.min())
                    and meta["raw_sha256"] == case["raw_sha256"] == array_sha(raw)
                    and meta["final_sha256"] == case["final_sha256"] == array_sha(final)
                    and meta["masked_target_logits_sha256"] == case["masked_target_logits_sha256"] == array_sha(masked)
                    and np.array_equal(postprocess(raw), final), f"Cache identity or content mismatch: {device}/{name}")
            full = np.asarray(case["independent_logits"]["original_logits"], dtype=np.float64)
            require(full.shape == (5,) and np.isfinite(full).all() and int(full.argmax()) == case["predicted_class"]
                    and np.isclose(full[0], meta["original_target_logit"], rtol=1e-5, atol=1e-6),
                    f"Independent original classification/logit evidence differs: {device}/{name}")
            loaded.append({"name": name, "device": device, "raw": raw, "final": final,
                           "masked": masked, "original": float(meta["original_target_logit"]),
                           "class": case["predicted_class"], "path": path, "key": key})
    drops = np.stack([item["original"] - item["masked"] for item in loaded])
    sum_occluded_drop = np.zeros((4, 3 * 224 * 224), dtype=np.float64)
    q_sum = np.zeros((3, 224, 224), dtype=np.float64)
    stream = hashlib.sha256()
    for start in range(0, COUNT, 32):
        up = F.interpolate(torch.from_numpy(np.ascontiguousarray(grids[start:start + 32], dtype=np.float32)),
                           size=(224, 224), mode="bilinear", align_corners=False).numpy()
        stream.update(np.ascontiguousarray(up).tobytes(order="C"))
        occluded = 1 - up.astype(np.float64)
        q_sum += occluded.sum(axis=0, dtype=np.float64)
        # Explicit einsum avoids the local Accelerate BLAS matmul warning flags
        # and independently sums the recorded finite values in CPU float64.
        block = np.einsum("ij,jk->ik", drops[:, start:start + 32],
                          occluded.reshape(32, -1), optimize=False)
        require(np.isfinite(block).all(), "Non-finite independent covariance block")
        sum_occluded_drop += block
    require(stream.hexdigest() == manifest["upsampled_stream_sha256"] and np.array_equal(q_sum / COUNT, q),
            "Independent CPU upsampling stream or full-bank q differs")
    reconstruction = []
    denominator = np.maximum(q * (1 - q), 1e-6)
    for index, item in enumerate(loaded):
        reconstructed = ((sum_occluded_drop[index].reshape(3, 224, 224) / COUNT
                          - q * drops[index].mean(dtype=np.float64)) / denominator).astype(np.float32)
        require(np.allclose(reconstructed, item["raw"][0], rtol=1e-5, atol=1e-6),
                f"Raw covariance cannot be independently reconstructed: {item['device']}/{item['name']}")
        reconstruction.append({"device": item["device"], "input": item["name"], "cache_key": item["key"],
                               "cache_sha256": sha(item["path"]),
                               "maximum_raw_reconstruction_abs_error": float(np.max(np.abs(reconstructed - item["raw"][0])))})
    existing = read_json(root / "candidate/probes/probe_comparison.json")
    require(existing["acceptance_sha256"] == ACCEPTANCE_SHA and existing["mask_bank_sha256"] == bank_identity,
            "Recorded candidate comparison uses a different acceptance or bank")
    results = {}
    for index, name in ((0, "original"), (2, "perturbed")):
        cpu, gpu = loaded[index:index + 2]
        cpu_logits = np.r_[cpu["original"], cpu["masked"]]
        gpu_logits = np.r_[gpu["original"], gpu["masked"]]
        close = np.isclose(cpu_logits, gpu_logits, rtol=limits["all_original_and_masked_target_logits"]["rtol"],
                           atol=limits["all_original_and_masked_target_logits"]["atol"])
        rel = float(np.linalg.norm(cpu["raw"].astype(np.float64) - gpu["raw"].astype(np.float64)) /
                    max(np.linalg.norm(cpu["raw"].astype(np.float64)), 1e-8))
        union = np.count_nonzero((cpu["final"] != 0) | (gpu["final"] != 0))
        intersection = np.count_nonzero((cpu["final"] != 0) & (gpu["final"] != 0))
        iou = float(intersection / union) if union else 1.0
        passed = bool(cpu["class"] == gpu["class"] and close.all() and rel <= limits["raw_relative_l2_max"]
                      and iou >= limits["final_nonzero_support_iou_min"])
        previous = existing["cases"][name]
        require(previous["pass"] == passed and previous["logit_fail_count"] == int((~close).sum())
                and previous["logits_compared"] == 1025
                and np.isclose(previous["raw_relative_l2"], rel, rtol=0, atol=1e-12)
                and previous["final_support_iou"] == iou,
                f"Recorded CPU/MPS candidate comparison differs: {name}")
        results[name] = {"pass": passed, "logits_compared": 1025, "logit_fail_count": int((~close).sum()),
                         "raw_relative_l2": rel, "final_support_iou": iou,
                         "predicted_class_match": cpu["class"] == gpu["class"]}
    status = "PASS" if all(item["pass"] for item in results.values()) else "FAIL"
    require(existing["status"] == status, "Recorded CPU/MPS comparison status differs")
    return {"status": status, "scope": "Independent read-only reconstruction of all 4096 saved masked logits",
            "acceptance_sha256": ACCEPTANCE_SHA, "preregistration_sha256": PREREG_SHA,
            "bank_identity_sha256": bank_identity, "raw_reconstruction": reconstruction, "cases": results}


if __name__ == "__main__":
    print(json.dumps(verify_candidate_probe(), ensure_ascii=False, indent=2))
