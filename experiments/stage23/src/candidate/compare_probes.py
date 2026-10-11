"""Independent CPU/MPS comparison of both fixed-input candidate probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import rise_cov as rc


ACCEPTANCE_SHA256 = "83c064cffdd1cc5993bac43a48dcfa4311d7b89ea1f4bd29f0454ff8aef4b5eb"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_case(probe: dict, name: str, bank: rc.MaskBank) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, float]:
    case = probe["cases"][name]
    path = Path(case["cache_path"])
    require(path.is_file(), f"Candidate cache missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(sorted(archive.files) == ["attribution", "masked_target_logits", "metadata", "raw_attribution"],
                f"Candidate cache schema differs: {name}")
        raw = np.asarray(archive["raw_attribution"])
        final = np.asarray(archive["attribution"])
        logits = np.asarray(archive["masked_target_logits"])
        meta = json.loads(str(archive["metadata"].item()))
    require(raw.shape == final.shape == (1,3,224,224)
            and raw.dtype == final.dtype == np.float32
            and logits.shape == (rc.MASK_COUNT,) and logits.dtype == np.float64,
            f"Candidate cache data shape or dtype differs: {name}")
    require(np.isfinite(raw).all() and np.isfinite(final).all() and np.isfinite(logits).all(),
            f"Non-finite candidate cache: {name}")
    require(case["raw_sha256"] == meta["raw_sha256"] == rc.array_sha256(raw)
            and case["final_sha256"] == meta["final_sha256"] == rc.array_sha256(final)
            and case["masked_target_logits_sha256"] == meta["masked_target_logits_sha256"] == rc.array_sha256(logits),
            f"Candidate cache content checksum differs: {name}")
    require(meta["key"] == case["cache_key"]
            and rc.sha256_bytes(rc.canonical_bytes(meta["identity"])) == meta["key"]
            and meta["identity"]["current_sample_sha256"] == case["input_sha256"]
            and meta["identity"]["callback_target_sha256"] == case["target_array_sha256"]
            and meta["device"] == probe["device"]
            and meta["precision"] == rc.PRECISION
            and meta["mask_bank_identity_sha256"] == bank.identity_sha256,
            f"Candidate cache identity or bank differs: {name}")
    require(np.array_equal(rc.postprocess_signed(raw).view(np.uint8), final.view(np.uint8)),
            f"Candidate postprocess differs: {name}")
    # Rebuild full covariance independently from the saved 1024 target logits.
    original = float(meta["original_target_logit"])
    drops = original - logits
    sum_od = np.zeros((3,224,224), dtype=np.float64)
    for start in range(0, rc.MASK_COUNT, 32):
        mask = bank.mask_batch(start, min(start+32, rc.MASK_COUNT))
        sum_od += np.einsum("n,nchw->chw", drops[start:start+len(mask)],
                            1.0-mask.astype(np.float64), dtype=np.float64, optimize=True)
    rebuilt = rc.covariance_from_sums(sum_od, float(drops.sum(dtype=np.float64)),
                                       bank.q, rc.MASK_COUNT).astype(np.float32)
    require(np.allclose(rebuilt, raw[0], rtol=1e-5, atol=1e-6),
            f"Candidate raw cannot be reconstructed from all masked logits: {name}")
    return case, raw, final, logits, original


def run(root: Path) -> dict:
    acceptance_path = root / "device/probe_acceptance.json"
    require(rc.sha256_file(acceptance_path) == ACCEPTANCE_SHA256,
            "Fixed device acceptance thresholds changed")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    limits = acceptance["correctness"]
    rtol = limits["all_original_and_masked_target_logits"]["rtol"]
    atol = limits["all_original_and_masked_target_logits"]["atol"]
    bank = rc.MaskBank(root / "candidate/bank")
    cpu = json.loads((root / "candidate/probes/probe_cpu.json").read_text(encoding="utf-8"))
    mps = json.loads((root / "candidate/probes/probe_mps.json").read_text(encoding="utf-8"))
    require(cpu["status"] == mps["status"] == "PASS"
            and cpu["device"] == "cpu" and mps["device"] == "mps",
            "Candidate CPU/MPS probes incomplete")
    require(cpu["mask_bank_sha256"] == mps["mask_bank_sha256"] == bank.identity_sha256,
            "CPU/MPS candidate mask bank differs")
    result_cases = {}
    for name in ("original", "perturbed"):
        c, c_raw, c_final, c_logits, c_original = load_case(cpu, name, bank)
        g, g_raw, g_final, g_logits, g_original = load_case(mps, name, bank)
        require(c["input_sha256"] == g["input_sha256"]
                and c["target_array_sha256"] == g["target_array_sha256"]
                and c["target_zero_based"] == g["target_zero_based"]
                and c["masked_count"] == g["masked_count"] == rc.MASK_COUNT,
                f"CPU/MPS candidate inputs or target differ: {name}")
        c_all = np.r_[c_original, c_logits]
        g_all = np.r_[g_original, g_logits]
        abs_error = np.abs(c_all - g_all)
        failed = ~np.isclose(c_all, g_all, rtol=rtol, atol=atol)
        norm_cpu = float(np.linalg.norm(c_raw.astype(np.float64)))
        raw_rel_l2 = float(np.linalg.norm(c_raw.astype(np.float64)-g_raw.astype(np.float64))
                           / max(norm_cpu, 1e-8))
        support_cpu = c_final != 0
        support_mps = g_final != 0
        union = int(np.count_nonzero(support_cpu | support_mps))
        intersection = int(np.count_nonzero(support_cpu & support_mps))
        iou = float(intersection / union) if union else 1.0
        sign_diff = int(np.count_nonzero((np.sign(c_raw) != np.sign(g_raw))
                                        & (c_raw != 0) & (g_raw != 0)))
        predicted_match = c["predicted_class"] == g["predicted_class"]
        passed = bool(predicted_match and not np.any(failed)
                      and raw_rel_l2 <= limits["raw_relative_l2_max"]
                      and iou >= limits["final_nonzero_support_iou_min"])
        result_cases[name] = {
            "pass": passed,
            "predicted_class_match": predicted_match,
            "predicted_class_cpu": c["predicted_class"],
            "predicted_class_mps": g["predicted_class"],
            "logits_compared": len(c_all),
            "logit_fail_count": int(np.count_nonzero(failed)),
            "logit_abs_error_max": float(abs_error.max()),
            "logit_abs_error_p50": float(np.percentile(abs_error, 50)),
            "logit_abs_error_p95": float(np.percentile(abs_error, 95)),
            "logit_abs_error_p99": float(np.percentile(abs_error, 99)),
            "raw_relative_l2": raw_rel_l2,
            "raw_abs_error_max": float(np.max(np.abs(c_raw-g_raw))),
            "raw_sign_disagreement_count": sign_diff,
            "final_support_iou": iou,
            "raw_reconstructible_from_all_1024_logits": True,
            "cpu_wall_seconds": c["wall_seconds"],
            "mps_wall_seconds": g["wall_seconds"],
            "speedup_cpu_over_mps": c["wall_seconds"] / g["wall_seconds"],
            "cpu_raw_sha256": c["raw_sha256"],
            "mps_raw_sha256": g["raw_sha256"],
            "cpu_final_sha256": c["final_sha256"],
            "mps_final_sha256": g["final_sha256"],
        }
    return {"status": "PASS" if all(case["pass"] for case in result_cases.values()) else "FAIL",
            "scope": "Stage23 engineering CPU/MPS candidate probe; not an official metric",
            "acceptance_sha256": ACCEPTANCE_SHA256,
            "method": rc.METHOD,
            "mask_bank_sha256": bank.identity_sha256,
            "cases": result_cases}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.root / "candidate/probes/probe_comparison.json"
    result = run(args.root)
    rc._atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
