"""Audit every Stage23 completed attribution archive, without model forwards."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

try:
    from .audit_candidate_probe import postprocess
except ImportError:
    from audit_candidate_probe import postprocess


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA = "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_all_complete_caches(root: Path = ROOT) -> dict:
    root = Path(root)
    began = time.monotonic()
    records: list[dict] = []
    excluded: list[dict] = []
    historical: list[dict] = []
    counts: Counter = Counter()
    npz_paths = sorted(root.rglob("*.npz"))
    reference_programs = {file_sha(root / "device/occlusion8_gpu.py"),
                          file_sha(root / "device/occlusion8_probe_frozen.py")}
    candidate_program = file_sha(root / "candidate/rise_cov.py")
    bank_manifest = json.loads((root / "candidate/bank/bank.json").read_text(encoding="utf-8"))
    bank_identity = hashlib.sha256(canonical(bank_manifest)).hexdigest()
    for path in npz_paths:
        relative = str(path.relative_to(root))
        if path == root / "device/probe_inputs.npz":
            excluded.append({"path": relative, "reason": "Frozen original/noise/perturbed inputs; not an attribution cache",
                             "sha256": file_sha(path)})
            continue
        if path.name.endswith(".partial.npz"):
            excluded.append({"path": relative, "reason": "Explicit incomplete safe-batch checkpoint; never counted as a complete attribution",
                             "bytes": path.stat().st_size, "sha256": file_sha(path)})
            continue
        with np.load(path, allow_pickle=False) as archive:
            fields = sorted(archive.files)
            metadata = json.loads(str(archive["metadata"].item()))
            if fields == ["attribution", "masked_target_logits", "metadata", "raw_attribution"]:
                kind = "candidate"
                raw = np.ascontiguousarray(archive["raw_attribution"])
                final = np.ascontiguousarray(archive["attribution"])
                masked = np.ascontiguousarray(archive["masked_target_logits"])
            elif fields == ["final", "masked_logits", "metadata", "raw"]:
                kind = "reference"
                raw = np.ascontiguousarray(archive["raw"])
                final = np.ascontiguousarray(archive["final"])
                masked = np.ascontiguousarray(archive["masked_logits"])
            else:
                raise RuntimeError(f"Unknown or incomplete attribution schema: {relative}: {fields}")
        require(raw.shape == final.shape == (1, 3, 224, 224)
                and raw.dtype == final.dtype == np.float32
                and np.isfinite(raw).all() and np.isfinite(final).all()
                and metadata["raw_sha256"] == array_sha(raw)
                and metadata["final_sha256"] == array_sha(final)
                and np.array_equal(postprocess(raw).view(np.uint8), final.view(np.uint8)),
                f"Raw/final shape, hash, finiteness or Stage22 postprocess differs: {relative}")
        key = metadata["key"]
        require(path.stem == key and path.parent.name == key[:2] and len(key) == 64,
                f"Cache path is not the content-addressed metadata key: {relative}")
        synthetic = "synthetic_cache" in path.parts
        if kind == "candidate" and "identity" not in metadata:
            require(relative == "candidate/synthetic_cache/rise_cov1024_fixed14_post22/1f/1fecd8f9e8567bb0beb2ba3a038cc04b9c6a959491a4482399f1dca1955ccbef.npz",
                    f"Unexpected cache missing current identity: {relative}")
            twin = root / "candidate/synthetic_cache/rise_cov1024_fixed14_post22/87/8785af3549d17e16e144fd149266c4397de6294cfa96b699e610217e24e6aebf.npz"
            with np.load(twin, allow_pickle=False) as archive:
                twin_meta = json.loads(str(archive["metadata"].item()))
                require(np.array_equal(raw.view(np.uint8), archive["raw_attribution"].view(np.uint8))
                        and np.array_equal(final.view(np.uint8), archive["attribution"].view(np.uint8))
                        and np.array_equal(masked.view(np.uint8), archive["masked_target_logits"].view(np.uint8)),
                        "Historical toy archive differs from current strictly validated synthetic twin")
            common_fields = ("baseline", "device", "precision", "input_sha256", "target_zero_based",
                             "original_target_logit", "mask_bank_identity_sha256", "raw_sha256",
                             "final_sha256", "masked_target_logits_sha256", "method_version")
            require(synthetic and all(metadata[field] == twin_meta[field] for field in common_fields)
                    and metadata["device"] == "cpu" and metadata["precision"] == "fp32"
                    and masked.shape == (1024,) and masked.dtype == np.float64 and np.isfinite(masked).all()
                    and metadata["masked_target_logits_sha256"] == array_sha(masked)
                    and metadata["mask_bank_identity_sha256"] == bank_identity
                    and metadata["spec_sha256"] == "a57615a369fc4cbaae3ac779b94be0984d472ed6ef8422d8acfb6f1fde4d7d1d",
                    "Historical toy content, metadata or recorded legacy spec digest differs")
            exception = {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha(path),
                         "classification": "historical_legacy_synthetic_archive",
                         "current_cache_identity_verified": False, "reusable_with_current_loader": False,
                         "reason": "Earlier toy cache lacks current callback identity and its old spec configuration is no longer available. Retained historical evidence; never used as a scored official-model result.",
                         "raw_final_logits_byte_identical_to_current_validated_twin": str(twin.relative_to(root)),
                         "raw_sha256": array_sha(raw), "final_sha256": array_sha(final),
                         "masked_logits_sha256": array_sha(masked), "masked_logits_count": 1024,
                         "device": "cpu", "weight_scope": "synthetic", "stage22_postprocess_byte_identical": True}
            historical.append(exception)
            counts[("candidate", "cpu", "historical_legacy_synthetic")] += 1
            records.append(exception)
            continue
        if kind == "candidate":
            identity = metadata["identity"]
            spec_digest = metadata["spec_sha256"]
            configs = []
            for config in path.parent.parent.glob("config_*.json"):
                spec = json.loads(config.read_text(encoding="utf-8"))
                if hashlib.sha256(canonical(spec)).hexdigest() == spec_digest:
                    configs.append(spec)
            require(len(configs) == 1, f"Candidate cache lacks exactly one matching configuration: {relative}")
            spec = configs[0]
            device = metadata["device"]
            target = metadata["target_zero_based"]
            input_digest = metadata["input_sha256"]
            target_dtype = np.dtype(identity["callback_target_dtype"])
            target_shape = identity["callback_target_shape"]
            require(target_dtype.kind in ("i", "u") and int(np.prod(target_shape)) == 1,
                    f"Candidate callback target shape/dtype differs from one integer label: {relative}")
            target_array = np.asarray([target], dtype=target_dtype).reshape(target_shape)
            require(masked.shape == (1024,) and masked.dtype == np.float64 and np.isfinite(masked).all()
                    and metadata["masked_target_logits_sha256"] == array_sha(masked)
                    and metadata["precision"] == spec["precision"] == "fp32"
                    and device == spec["device"] and device in ("cpu", "mps", "cuda")
                    and spec["method"] == "rise_cov1024_fixed14_post22" and spec["mask_count"] == 1024
                    and spec["rise_cov_code_sha256"] == candidate_program
                    and metadata["mask_bank_identity_sha256"] == spec["mask_bank_identity_sha256"] == bank_identity
                    and hashlib.sha256(canonical(identity)).hexdigest() == key
                    and identity["spec_sha256"] == spec_digest
                    and identity["current_sample_sha256"] == input_digest
                    and identity["current_sample_shape"] == [3, 224, 224]
                    and identity["current_sample_dtype"] == "<f4"
                    and identity["callback_input_shape"] == [1, 3, 224, 224]
                    and identity["callback_input_dtype"] == "<f4"
                    and identity["callback_input_sha256"] == input_digest
                    and identity["target_zero_based"] == target and 0 <= target <= 4
                    and identity["callback_target_sha256"] == array_sha(target_array)
                    and np.isfinite(metadata["baseline"]) and np.isfinite(metadata["original_target_logit"]),
                    f"Candidate masked logits/device/identity mismatch: {relative}")
            require(spec["weight_sha256"] == ("a" * 64 if synthetic else WEIGHT_SHA),
                    f"Candidate actual/synthetic weight identity differs: {relative}")
            method = spec["method"]
        else:
            spec = metadata["spec"]
            record = metadata["record"]
            device = spec["device"]
            target = metadata["target"]
            input_digest = metadata["input_sha256"]
            identity = {"spec": spec, "input_shape": [3, 224, 224], "input_dtype": "<f4",
                        "input_sha256": input_digest, "target_dtype": "<i8", "target": target}
            full_logits = np.asarray(record["original_logits"], dtype=np.float64)
            require(masked.shape == (2352,) and masked.dtype == np.float32 and np.isfinite(masked).all()
                    and metadata["masked_logits_sha256"] == array_sha(masked)
                    and spec["method"] == "black_occlusion8_signed"
                    and spec["tile_size"] == 8 and spec["tile_count"] == record["mask_count"] == 2352
                    and spec["batch_size"] == record["batch_size"] and spec["batch_size"] in (32, 64)
                    and spec["precision"] == record["precision"] == "float32"
                    and spec["program_sha256"] in reference_programs
                    and spec["weight_sha256"] == WEIGHT_SHA
                    and device == record["device"] and device in ("cpu", "mps", "cuda")
                    and record["input_sha256"] == input_digest and record["target"] == target and 0 <= target <= 4
                    and hashlib.sha256(canonical(identity)).hexdigest() == key
                    and full_logits.shape == (5,) and np.isfinite(full_logits).all()
                    and record["predicted_class"] == int(full_logits.argmax())
                    and record["original_target_logit"] == full_logits[target]
                    and np.isfinite(record["baseline"]), f"Reference masked logits/device/identity mismatch: {relative}")
            tile_values = ((float(record["original_target_logit"]) - masked.astype(np.float64)) / 64).astype(np.float32)
            reconstructed = np.repeat(np.repeat(tile_values.reshape(3, 28, 28), 8, axis=1), 8, axis=2)
            require(np.array_equal(reconstructed.view(np.uint8), raw[0].view(np.uint8)),
                    f"Reference raw cannot be reproduced from all 2352 saved masked logits: {relative}")
            method = spec["method"]
        counts[(kind, device, "synthetic" if synthetic else "official_weights")] += 1
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha(path),
                        "schema": kind, "method": method, "device": device,
                        "weight_scope": "synthetic" if synthetic else "official_weights",
                        "cache_key": key, "current_input_sha256": input_digest,
                        "raw_sha256": array_sha(raw), "final_sha256": array_sha(final),
                        "masked_logits_sha256": array_sha(masked), "masked_logits_count": len(masked),
                        "stage22_postprocess_byte_identical": True,
                        "current_cache_identity_verified": True,
                        "reference_raw_reconstructed_from_saved_logits": kind == "reference"})
    require(len(historical) <= 1, "More than the known single historical synthetic archive exception")
    return {"status": "PASS", "scope": "All recursively discovered Stage23 npz attribution archive content; current cache identities except one explicitly retained historical toy archive",
            "new_model_forwards": 0, "rewritten_cache_files": 0,
            "elapsed_seconds": time.monotonic() - began,
            "discovered_npz_count": len(npz_paths), "complete_cache_count": len(records),
            "current_compatible_cache_count": len(records) - len(historical),
            "historical_legacy_count": len(historical),
            "official_weight_cache_identity_exception_count": 0,
            "historical_legacy": historical,
            "excluded_count": len(excluded),
            "partial_checkpoint_count": sum(path["reason"].startswith("Explicit incomplete") for path in excluded),
            "counts": [{"schema": key[0], "device": key[1], "weight_scope": key[2], "count": count}
                       for key, count in sorted(counts.items())],
            "excluded": excluded, "entries": records}


if __name__ == "__main__":
    torch.set_num_threads(4)
    result = verify_all_complete_caches()
    destination = ROOT / "audit/all_complete_caches.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "entries"}, ensure_ascii=False, indent=2))
