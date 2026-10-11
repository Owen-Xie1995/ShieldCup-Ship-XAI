"""Hand arithmetic, bank reproducibility and full synthetic estimator checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

import rise_cov as rc


def run(bank_dir: Path) -> dict:
    began = time.monotonic()
    # Four hand-picked drops; the first feature's occlusion is positively
    # associated with logit loss and the second is negatively associated.
    drops = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    occluded = np.array([[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=np.float64)
    q = occluded.mean(axis=0)
    observed = rc.covariance_from_sums(np.sum(occluded * drops[:, None], axis=0),
                                        float(drops.sum()), q, 4)
    hand = np.array([4.0, -2.0, 0.0], dtype=np.float64)
    # For the second column, q=.5 and E[oD]=(1+5)/4=1.5;
    # E[D]=4, hence (1.5-2)/.25=-2.
    if not np.array_equal(observed, hand):
        raise AssertionError(f"Hand signed covariance mismatch: {observed} vs {hand}")

    torch.set_num_threads(8)
    bank32 = rc.MaskBank(bank_dir, generation_batch_size=32)
    bank64 = rc.MaskBank(bank_dir, generation_batch_size=64)
    if bank32.stream_sha256 != bank64.stream_sha256 or not np.array_equal(bank32.q, bank64.q):
        raise AssertionError("Bank stream or q changes with generation batch size")
    if np.any(bank32.q <= 0) or np.any(bank32.q >= 1):
        raise AssertionError("Unexpected q saturation in fixed Bernoulli bank")
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(1, 3, 224, 224)).astype(np.float32)
    stage22_path = Path(__file__).resolve().parents[2] / "output_mac_stage22/experiment.py"
    spec = importlib.util.spec_from_file_location("stage22_postprocess_for_test", stage22_path)
    stage22 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(stage22)
    p22 = stage22.postprocess_signed(raw)
    p23 = rc.postprocess_signed(raw)
    if not np.array_equal(p22.view(np.uint8), p23.view(np.uint8)):
        raise AssertionError("Stage 23 postprocessing differs bytewise from Stage 22")

    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

        def forward(self, x):
            z0 = (x[:, 0].mean(dim=(1, 2)) + 0.7 * x[:, 1].mean(dim=(1, 2))
                  - 0.4 * x[:, 2].mean(dim=(1, 2)) + self.bias)
            return torch.stack((z0, -z0), dim=1)

    model = SmallModel().eval()
    x = rng.normal(size=(3, 224, 224)).astype(np.float32)
    cache = rc.RiseCache(Path(bank_dir).parent / "synthetic_cache", bank32,
                         weight_sha256="a" * 64, device="cpu", inference_batch_size=32)
    key = cache.key(x, 0)
    perturbed = np.ascontiguousarray(x + np.float32(1e-3))
    if cache.key(perturbed, 0) == key:
        raise AssertionError("Perturbed full current input reused original cache key")
    if cache.key(x, 0, x[None], np.array([0], dtype=np.int32)) == key:
        raise AssertionError("Callback target dtype is missing from cache key")
    if cache.key(x, 0, x[None], np.array([[0]], dtype=np.int64)) == key:
        raise AssertionError("Callback target shape is missing from cache key")
    raw1, final1 = rc.explain_pair(model, x[None], np.array([0], dtype=np.int64),
                                    device="cpu", inference_batch_size=32,
                                    bank=bank32, cache=cache)
    stored = cache.load(key)
    if stored is None:
        raise AssertionError("Full synthetic attribution was not cached")
    if cache.partial_path(key).exists():
        raise AssertionError("Completed cache retained obsolete partial state")
    with np.load(cache.path(key), allow_pickle=False) as archive:
        logits = np.asarray(archive["masked_target_logits"])
        meta = json.loads(str(archive["metadata"].item()))
    original_logit = float(meta["original_target_logit"])
    baseline = np.float32(x.min())
    for index in (0, rc.MASK_COUNT - 1):
        mask = bank32.mask_batch(index, index + 1)
        masked = torch.tensor(baseline) + (torch.from_numpy(x[None]) - baseline) * torch.from_numpy(mask)
        with torch.no_grad():
            direct = float(model(masked)[0, 0])
        if not np.isclose(direct, logits[index], rtol=0, atol=2e-6):
            raise AssertionError(f"Mask {index} target logit mismatch")
    # Independent direct formula on selected pixels, using all 1024 stored D_j.
    d = original_logit - logits
    checked_pixels = [(0, 0, 0), (1, 80, 100), (2, 223, 223)]
    checked_values = []
    for ch, row, col in checked_pixels:
        o = np.empty(rc.MASK_COUNT, dtype=np.float64)
        for start in range(0, rc.MASK_COUNT, 64):
            m = bank32.mask_batch(start, min(start + 64, rc.MASK_COUNT))
            o[start:start + len(m)] = 1.0 - m[:, ch, row, col]
        q_pixel = float(np.mean(o))
        direct = float(np.mean((o - q_pixel) * d) / max(q_pixel * (1 - q_pixel), 1e-6))
        observed_pixel = float(raw1[0, ch, row, col])
        if not np.isclose(direct, observed_pixel, rtol=1e-5, atol=1e-6):
            raise AssertionError(f"Direct covariance mismatch at {(ch,row,col)}")
        checked_values.append({"index": [ch, row, col], "direct": direct,
                               "observed": observed_pixel})
    if not np.isfinite(raw1).all() or not np.isfinite(final1).all():
        raise AssertionError("Synthetic output is not finite")
    return {
        "status": "PASS",
        "hand_covariance_expected": hand.tolist(),
        "hand_covariance_observed": observed.tolist(),
        "mask_bank_sha256": bank32.identity_sha256,
        "mask_stream_sha256": bank32.stream_sha256,
        "q_sha256": bank32.q_array_sha256,
        "batch32_batch64_bank_identity": True,
        "stage22_postprocess_byte_identical": True,
        "synthetic_full_mask_count": rc.MASK_COUNT,
        "synthetic_input_sha256": rc.array_sha256(x),
        "synthetic_cache_key": key,
        "perturbed_cache_key_distinct": True,
        "callback_target_dtype_and_shape_bound_in_key": True,
        "completed_cache_cleared_partial": True,
        "synthetic_raw_sha256": rc.array_sha256(raw1),
        "synthetic_final_sha256": rc.array_sha256(final1),
        "synthetic_masked_logits_sha256": rc.array_sha256(logits),
        "independently_checked_pixels": checked_values,
        "elapsed_seconds": time.monotonic() - began,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path,
                        default=Path(__file__).resolve().parent / "bank")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "test_result.json")
    args = parser.parse_args()
    result = run(args.bank_dir)
    rc._atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
