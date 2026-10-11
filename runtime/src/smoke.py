"""Verify deployment sources and a deterministic toy explanation without ResNet forward."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from explainer import MASK_BATCH_SIZE, METHOD, custom_explain_func
from runtime_support import MODEL_ROOT, atomic_json, environment, selected_device, sha, verify_sources


class LiteralLinearLogits(torch.nn.Module):
    """Five fixed linear logits. Construction never calls a random initializer."""

    def __init__(self):
        super().__init__()
        channel_weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        class_weights = np.asarray([1.0, -2.0, -0.5, 0.25, 0.0], dtype=np.float32)
        weights = np.broadcast_to(
            class_weights[:, None, None, None] * channel_weights[None, :, None, None],
            (5, 3, 8, 8),
        ).copy()
        self.register_buffer("weights", torch.from_numpy(weights))
        self.register_buffer(
            "bias", torch.tensor([0.0, 2.0, 3.0, 4.0, 1_000_000.0], dtype=torch.float32)
        )
        self.calls: list[dict] = []

    def forward(self, inputs):
        self.calls.append({"batch": len(inputs), "device": str(inputs.device),
                           "dtype": str(inputs.dtype), "eval": not self.training})
        values = (inputs[:, None] * self.weights[None]).flatten(2).sum(dim=2)
        return values + self.bias


def literal_inputs():
    # Each literal is a 2x2 array of 4x4 tile values, repeated in three channels.
    tiles = np.asarray([[[-2.0, -1.0], [1.0, 3.0]],
                        [[-3.0, -0.5], [0.5, 2.0]]], dtype=np.float32)
    spatial = np.repeat(np.repeat(tiles, 4, axis=1), 4, axis=2)
    return np.repeat(spatial[:, None], 3, axis=1), np.asarray([0, 2], dtype=np.int64)


def rng_snapshot(device):
    state = np.random.get_state()
    result = {"python": random.getstate(),
              "numpy": (state[0], state[1].copy(), *state[2:]),
              "torch_cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        result["torch_cuda"] = torch.cuda.get_rng_state(device).clone()
    elif device.type == "mps":
        result["torch_mps"] = torch.mps.get_rng_state().clone()
    return result


def compare_rng(before, after):
    same = {"python": before["python"] == after["python"],
            "numpy": before["numpy"][0] == after["numpy"][0]
            and np.array_equal(before["numpy"][1], after["numpy"][1])
            and before["numpy"][2:] == after["numpy"][2:]}
    for name in before:
        if name.startswith("torch_"):
            same[name] = bool(torch.equal(before[name], after[name]))
    return same


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_toy(device):
    inputs, targets = literal_inputs()
    input_before, target_before = inputs.tobytes(), targets.tobytes()
    before = rng_snapshot(device)
    model = LiteralLinearLogits().train()
    traces = []
    output = custom_explain_func(model, inputs, targets, device, trace_hook=traces.append)
    unchanged = compare_rng(before, rng_snapshot(device))
    checks = {
        "shape": output.shape == inputs.shape == (2, 3, 8, 8),
        "finite_float32": output.dtype == np.float32 and bool(np.isfinite(output).all()),
        "input_unchanged": inputs.tobytes() == input_before,
        "target_unchanged": targets.tobytes() == target_before,
        "python_numpy_torch_rng_unchanged": all(unchanged.values()),
        "positive_sample0": bool((output[0] > 0).any()),
        "negative_sample1": bool((output[1] < 0).any()),
        "two_sample_traces": len(traces) == 2,
        "eval_fp32": not model.training and all(
            call["eval"] and call["dtype"] == "torch.float32" for call in model.calls),
        "two_original_and_two_mask_batches": [call["batch"] for call in model.calls]
        == [1, 12, 1, 12],
    }
    for name, passed in checks.items():
        require(passed, f"Toy smoke check failed: {name}")
    sample_records = []
    for sample_index, trace in enumerate(traces):
        target = int(targets[sample_index])
        predicted = int(trace["logits"].argmax(axis=1)[0])
        require(predicted == 4 and target != predicted, "Supplied target must differ from argmax")
        require(trace["target_index"] == target, "Explanation changed the supplied target")
        require(trace["logits"].shape == (1, 5), "Original logits shape differs")
        require(trace["masked_logits"].shape == (12, 5), "Masked logits shape differs")
        require(trace["mask_batch_size"] == 64 and trace["tile_size"] == 4,
                "Frozen mask configuration differs")
        require(trace["device"] == str(model.weights.device), "The model used another device")
        # The literal inputs and per-channel weights are constant inside each tile.
        # Thus this linear toy has an exact analytic raw explanation for supplied targets.
        weights = model.weights[target].detach().cpu().numpy()
        expected_raw = ((inputs[sample_index] - inputs[sample_index].min()) * weights)[None]
        require(np.array_equal(trace["raw"], expected_raw),
                "Signed raw differs from the analytic supplied-target intervention")
        sample_records.append({
            "sample_index": sample_index, "supplied_target": target, "argmax": predicted,
            "masks": 12, "baseline": trace["baseline"],
            "raw_matches_analytic_supplied_target": True,
            "positive_final_values": int((output[sample_index] > 0).sum()),
            "negative_final_values": int((output[sample_index] < 0).sum()),
        })
    checks.update(supplied_target_not_argmax=True, signed_raw_analytic_exact=True,
                  explicit_device=True, fixed_mask_batch64=True)
    return {"checks": checks, "rng_unchanged": unchanged, "samples": sample_records,
            "input_shape": list(inputs.shape), "output_shape": list(output.shape),
            "dtype": str(output.dtype), "toy_forward_count": len(model.calls),
            "toy_forwards": model.calls,
            "output_sha256": hashlib.sha256(output.tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, help="Optional JSON smoke result")
    args = parser.parse_args()
    started = time.monotonic()
    record = {"status": "RUNNING", "method": METHOD, "requested_device": args.device,
              "official_model_loaded": False, "official_model_forward_count": 0,
              "mask_batch_size": MASK_BATCH_SIZE,
              "scope": "Source identities and a literal linear toy; no official model inference"}
    try:
        record["source_binding"] = verify_sources(MODEL_ROOT / "checkpoints/best_resnet50.pth")
        device = selected_device(args.device)
        record.update(environment=environment(), device=str(device), precision="float32",
                      explainer_sha256=sha(Path(__file__).with_name("explainer.py")),
                      smoke_sha256=sha(Path(__file__)))
        record["inside_container"] = record["environment"]["inside_container"]
        record.update(run_toy(device))
        record.update(status="PASS", elapsed_seconds=time.monotonic() - started)
    except Exception as exc:
        record.update(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.monotonic() - started)
        if args.output is not None:
            atomic_json(args.output, record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        raise
    if args.output is not None:
        atomic_json(args.output, record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
