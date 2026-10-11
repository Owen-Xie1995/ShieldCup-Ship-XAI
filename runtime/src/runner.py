"""Use the unmodified official wheel to generate a new deployment CSV."""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
from explainer import custom_explain_func, METHOD, MASK_BATCH_SIZE
from runtime_support import MODEL_ROOT, WEIGHT_SHA, atomic_json, environment, model_state_digest, official_device, selected_device, sha, verify_sources

FIELDS = ["Id", "Faithfulness", "Max_Sensitivity", "Sparsity"]

class BoundedExplainer:
    """Only current-run final outputs; no precomputed inputs, raw or disk cache."""
    def __init__(self, capacity=6):
        if not 0 <= capacity <= 32: raise ValueError("Runtime cache capacity must be 0..32")
        self.capacity = capacity; self.cache = OrderedDict(); self.hits = 0; self.cold = 0
        self.identity_seconds = 0.0

    def __call__(self, model, inputs, targets, device, **kwargs):
        requested = selected_device(device)
        model.to(device=requested, dtype=torch.float32).eval()
        use_cache = bool(self.capacity) and kwargs.get("trace_hook") is None
        key = None
        if use_cache:
            identity_started = time.monotonic()
            x = np.ascontiguousarray(inputs); y = np.ascontiguousarray(targets)
            key = (id(model), model_state_digest(model), WEIGHT_SHA, str(requested),
                   "float32", METHOD, MASK_BATCH_SIZE,
                   x.shape, x.dtype.str, hashlib.sha256(x.tobytes()).digest(),
                   y.shape, y.dtype.str, hashlib.sha256(y.tobytes()).digest())
            self.identity_seconds += time.monotonic() - identity_started
        if use_cache and key in self.cache:
            self.hits += 1; self.cache.move_to_end(key)
            print(json.dumps({"runtime_cache": "hit", "entries": len(self.cache), "method": METHOD}), flush=True)
            return self.cache[key].copy()
        self.cold += 1
        def progress(record): print(json.dumps({"explainer_progress": record}), flush=True)
        kwargs.pop("progress", None)
        output = custom_explain_func(model, inputs, targets, device, progress=progress, **kwargs)
        if use_cache:
            self.cache[key] = output.copy()
            while len(self.cache) > self.capacity: self.cache.popitem(last=False)
        return output

class Tee:
    def __init__(self, file): self.file = file; self.nan_warning = False
    def write(self, value):
        if "NaN 空值" in value or "自動補 0.0" in value: self.nan_warning = True
        sys.__stdout__.write(value); sys.__stdout__.flush(); self.file.write(value); self.file.flush()
        return len(value)
    def flush(self): self.file.flush(); sys.__stdout__.flush()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-file", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output CSV; existing files are refused")
    parser.add_argument("--weights", type=Path, default=MODEL_ROOT / "checkpoints/best_resnet50.pth")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Our evaluation seed; not an official mandatory seed")
    parser.add_argument("--cache-capacity", type=int, default=6)
    args = parser.parse_args()
    pending = args.output.with_name(args.output.name + ".pending")
    invalid = args.output.with_name(args.output.name + ".invalid")
    logfile = args.output.with_suffix(".stdout.log")
    runfile = args.output.with_suffix(".run.json")
    for artifact in (args.output, pending, invalid, logfile, runfile):
        if artifact.exists():
            raise FileExistsError(f"Choose a new output stem; existing CSV or run evidence is protected: {artifact}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = selected_device(args.device)
    if device.type != official_device():
        raise RuntimeError("Requested device differs from the unchanged generator's auto selection; control visible GPUs externally")
    binding = verify_sources(args.weights)
    from kaggle.model.model import load_model
    from kaggle.model.dataset.dataloader import get_dataloader
    from kaggle.model.dataset.transforms import val_transforms
    from kaggle.submmit.submmit import generate_xai_submission
    from official_metrix import score
    model = load_model(str(args.weights), device=torch.device("cpu"))
    model.to(device=device, dtype=torch.float32).eval()
    loader = get_dataloader(str(args.csv_file), str(args.images_dir), batch_size=1, num_workers=0, shuffle=False)
    # Explicit official validation transforms match the frozen Mac experiments.
    # The official loader's default train_transforms includes random flips.
    loader.dataset.transform = val_transforms
    callback = BoundedExplainer(args.cache_capacity)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    image_sources = []
    for _, row in loader.dataset.annotations.iterrows():
        image = args.images_dir / str(row["image"])
        image_sources.append({"image": str(row["image"]), "target": int(row["category"]) - 1,
                              "bytes": image.stat().st_size, "sha256": sha(image)})
    record = {"status": "RUNNING", "method": METHOD, "seed": args.seed, "device": str(device),
              "elapsed_seconds_scope": "Monotonic clock starts in main after initial module imports, CLI parsing, and output-path preflight; includes device/source/weight checks, lazy official imports, model loading, image IO/hash, scoring and runtime-cache identity hashing. Excludes Python process/module startup; not the complete cold-start or MPS regression budget clock.",
              "precision": "float32", "mask_batch_size": 64, "source_binding": binding,
              "environment": environment(), "input_csv_sha256": sha(args.csv_file),
              "input_csv": str(args.csv_file.resolve()), "images_dir": str(args.images_dir.resolve()),
              "input_images": image_sources,
              "preprocessing": "Official val_transforms, batch1, shuffleFalse, workers0; our alignment settings",
              "explainer_sha256": sha(Path(__file__).with_name("explainer.py")),
              "no_historical_cache_dependency": True}
    # Exclusive creation reserves this output stem for this run. Updates below
    # replace only our own record, and another runner cannot take the same stem.
    with runfile.open("x") as runlog:
        runlog.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    owns_pending = False
    try:
        import contextlib
        with pending.open("x"): pass
        owns_pending = True
        with logfile.open("x") as log:
            tee = Tee(log)
            with contextlib.redirect_stdout(tee):
                frame = generate_xai_submission(model, loader, callback, output_path=str(pending))
        if tee.nan_warning: raise RuntimeError("Official generator reported NaN replacement; CSV is not valid evidence")
        saved = pd.read_csv(pending)
        if saved.columns.tolist() != FIELDS or saved.Id.tolist() != list(range(1, len(loader.dataset) + 1)):
            raise RuntimeError("Official output CSV schema or row order differs")
        if not np.isfinite(saved[FIELDS[1:]].to_numpy(float)).all(): raise RuntimeError("Nonfinite official metrics")
        values = 20 * (1 + saved.Faithfulness.to_numpy(float)) + 30 * np.maximum(1 - saved.Max_Sensitivity.to_numpy(float) / 2, 0) + 30 * saved.Sparsity.to_numpy(float)
        official_score = float(score(pd.DataFrame({"Id": saved.Id}), saved.copy(), "Id"))
        if not np.isclose(official_score, values.mean(), atol=1e-9, rtol=0): raise RuntimeError("Official scorer differs from row formula")
        # Publish the already validated bytes without replacing a concurrently
        # created CSV. Pending and final paths share the same filesystem.
        os.link(pending, args.output)
        pending.unlink()
        owns_pending = False
        record.update(status="COMPLETE", samples=len(saved), official_score=official_score,
                      metric_means={name:float(saved[name].mean()) for name in FIELDS[1:]},
                      csv_sha256=sha(args.output), elapsed_seconds=time.monotonic()-started,
                      cache={"capacity":callback.capacity,"cold_inputs":callback.cold,"hits":callback.hits,
                             "identity_seconds":callback.identity_seconds},
                      scope="Actual rows in the supplied test CSV; no undisclosed-set claim")
        atomic_json(runfile, record); print(json.dumps(record, ensure_ascii=False, indent=2))
    except BaseException as exc:
        record.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
        if owns_pending and pending.exists():
            try:
                os.link(pending, invalid); pending.unlink()
                record["invalid_csv"] = str(invalid.resolve())
            except OSError as preserve_error:
                record["pending_checkpoint"] = str(pending.resolve())
                record["preservation_error"] = f"{type(preserve_error).__name__}: {preserve_error}"
        atomic_json(runfile, record)
        raise

if __name__ == "__main__": main()
