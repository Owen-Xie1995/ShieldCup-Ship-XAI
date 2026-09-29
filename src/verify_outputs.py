"""Independently audit Stage 22 sources, scores, cache, heatmaps and paired branch.

Run after the official nine-image experiment, both-method heatmaps, observations,
branch decision, and any required paired repeats are complete. No model forward
passes are needed; only the organizer's unmodified metrix.py is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CONTROL = "control"
PREVIOUS = "black_occlusion16_signed"
CANDIDATE = "black_occlusion8_signed"
METRICS = ("Faithfulness", "Max_Sensitivity", "Sparsity")
TERMS = ("faithfulness_contribution", "sensitivity_contribution", "sparsity_contribution")
COLUMNS = ["Id", *METRICS]
SCORE_42 = {CONTROL: 75.58758067203142, PREVIOUS: 76.05259533915773}
FIXED_HASHES = {
    "weights": "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473",
    "official_generator": "3f0c8830e590f6ad46934c034a5bf75f2d2b1d89baf2dba6683cdc1d9dc61df1",
    "official_scorer": "5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda",
    "prior_control_csv": "2e177fff03991c529767393b4aceb796cc380d7a724d6c0e31d5fbe829cfe196",
    "prior_checkpoint_zip": "66f7b405136f6cbb1a17c00c83ccff0d9a2801aeadc3fef58a79bb19af9850cb",
    "stage20_checkpoint_zip": "416927931fa74d5c0deec4282e333c03d124eeadf7dead0b6eccde8d9dd76dc2",
    "stage20_experiment": "3cb6ec81c0214012ff85833020a0f2fddee175b452c492412129a470cdcd08fd",
    "stage21_checkpoint_zip": "9b36d2b568019da9809ef534d181f266dace0f042ede37ecdf0dbdfae57a7fee",
    "stage21_experiment": "5264364073961e5c3f8de1e1982fe2856de4eeecca40aca4ca5ca459784360f5",
    "stage21_csv": "c63544390619d5b83b85d2c81ac356985364c9235c254998ac34efb0eae3d267",
    "stage22_code/experiment.py": "0481ee5206da6cdb0f419975f25a091f9ea5d43a0658593996f2de03f88f4df2",
}
REQUIRED_SOURCES = {
    "weights", "official_generator", "official_scorer", "official_score_wrapper",
    "official_solution", "official_test_csv", "prior_control_csv",
    "prior_control_run", "prior_control_experiment", "stage21_csv",
    "stage21_run", "stage21_score", "stage21_cache_config",
    "stage22_code/experiment.py", "stage22_code/probe.py",
    "stage22_code/heatmaps.py", "stage22_code/summarize.py",
    "stage22_code/verify_outputs.py", "stage22/probe.json",
}
EXPECTED_METHODS = ["stage21_16x16", "stage22_8x8"]
EXPECTED_SHAPE = (1, 3, 224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def close(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    require(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tol,
            f"{label}: {actual} != {expected}")


def same_column(frame: pd.DataFrame, name: str, expected: np.ndarray) -> None:
    require(name in frame.columns, f"Comparison column missing: {name}")
    got = frame[name].to_numpy(dtype=float)
    require(got.shape == expected.shape and np.allclose(got, expected, atol=1e-12, rtol=0),
            f"Comparison column differs: {name}")


def read_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require(frame.columns.tolist() == COLUMNS and frame.Id.tolist() == list(range(1, 10)),
            f"Invalid official nine-image CSV: {path}")
    require(np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all(),
            f"Non-finite official metric: {path}")
    return frame


def row_parts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        TERMS[0]: 20.0 * (1.0 + frame.Faithfulness.to_numpy(dtype=float)),
        TERMS[1]: 30.0 * np.maximum(
            1.0 - frame.Max_Sensitivity.to_numpy(dtype=float) / 2.0, 0.0
        ),
        TERMS[2]: 30.0 * frame.Sparsity.to_numpy(dtype=float),
    }


def environment_sources(root: Path) -> tuple[dict, dict[str, Path]]:
    environment = load_json(root / "environment.json")
    records = environment["files"]
    require(REQUIRED_SOURCES.issubset(records), "Environment source inventory incomplete")
    paths: dict[str, Path] = {}
    for name, record in records.items():
        path = Path(record["path"])
        require(path.is_file() and path.stat().st_size == record["bytes"],
                f"Environment source missing or byte count changed: {name}: {path}")
        require(sha256_file(path) == record["sha256"],
                f"Environment source SHA-256 changed: {name}: {path}")
        paths[name] = path
    for name, expected in FIXED_HASHES.items():
        require(records[name]["sha256"] == expected, f"Unexpected fixed source hash: {name}")
    require(records["weights"]["bytes"] == 282678575, "Official weight byte count changed")
    require(records["stage20_checkpoint_zip"]["bytes"] == 10083094,
            "Stage 20 ZIP byte count changed")
    require(records["stage21_checkpoint_zip"]["bytes"] == 510091,
            "Stage 21 ZIP byte count changed")
    test = pd.read_csv(paths["official_test_csv"])
    require(test.columns.tolist() == ["image", "category"] and len(test) == 9
            and not test.image.duplicated().any(), "Official test-list schema changed")
    for filename in test.image:
        require(f"test_image/{filename}" in records,
                f"Official image missing from environment records: {filename}")
    require(environment.get("test_order") == test[["image", "category"]].to_dict(orient="records"),
            "Environment test order differs from official CSV")
    return environment, paths


def official_score_function(path: Path):
    spec = importlib.util.spec_from_file_location("stage22_original_metrix", path)
    require(spec is not None and spec.loader is not None, "Cannot import original metrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score


def verify_scored_run(
    csv_path: Path, score_path: Path, run_path: Path, method: str, seed: int,
    scorer, solution: pd.DataFrame, records: dict, experiment_path: Path,
) -> tuple[dict, pd.DataFrame, dict]:
    frame = read_metrics(csv_path)
    score = load_json(score_path)
    run = load_json(run_path)
    digest = sha256_file(csv_path)
    require(score.get("status") == "C3_SAMPLE_SCORER_PASS"
            and score.get("sample_count") == 9
            and score.get("column_order") == COLUMNS
            and score.get("row_ids") == list(range(1, 10)),
            f"Original scorer status/schema failed: {csv_path}")
    require(score.get("submission_sha256") == digest
            and run.get("submission_csv_sha256") == digest,
            f"Scored CSV digest mismatch: {csv_path}")
    require(score.get("scorer_sha256") == records["official_scorer"]["sha256"]
            and score.get("solution_sha256") == records["official_solution"]["sha256"],
            f"Scorer or solution hash mismatch: {csv_path}")
    require(run.get("status") == "OFFICIAL_GENERATOR_AND_FORMULA_NINE_IMAGE_RESULT"
            and run.get("evaluation_seed") == seed
            and run.get("official_weight_sha256") == FIXED_HASHES["weights"]
            and run.get("attribution_is_dynamic") is True
            and run.get("batch_size") == 1 and run.get("shuffle") is False
            and run.get("device") == "cpu",
            f"Scored run identity/configuration mismatch: {run_path}")
    expected_method = ("control_ig50_smooth_abs_top25_mean5_top10"
                       if method == CONTROL else method)
    require(run.get("method") == expected_method, f"Method mismatch: {run_path}")
    hashes = run["code_sha256"]
    for field, source in (("official_generator", "official_generator"),
                          ("official_scorer", "official_scorer"),
                          ("official_wrapper", "official_score_wrapper")):
        require(hashes.get(field) == records[source]["sha256"],
                f"Run source hash mismatch: {run_path}/{field}")
    require(hashes.get("experiment") == sha256_file(experiment_path),
            f"Run experiment hash mismatch: {run_path}")
    score_value = float(scorer(solution, frame, "Id"))
    parts = row_parts(frame)
    expanded = float(sum(values.mean() for values in parts.values()))
    close(float(score["official_score_0_to_100"]), score_value, f"Original scorer {method}/{seed}")
    close(float(run["official_formula_score"]), score_value, f"Run formula {method}/{seed}")
    close(expanded, score_value, f"Expanded formula {method}/{seed}")
    for metric in METRICS:
        avg = float(frame[metric].mean())
        close(run["metric_means"][metric], avg, f"Run metric mean {method}/{seed}/{metric}")
        close(score["metric_summary"][metric]["mean"], avg,
              f"Scorer metric mean {method}/{seed}/{metric}")
    for term, values in parts.items():
        name = term.removesuffix("_contribution")
        close(run["score_contributions"][name], float(values.mean()),
              f"Run score contribution {method}/{seed}/{name}")
    return {
        "score": score_value,
        "csv_sha256": digest,
        "means": {metric: float(frame[metric].mean()) for metric in METRICS},
        "contributions": {term: float(values.mean()) for term, values in parts.items()},
    }, frame, run


def verify_summary(root: Path, frames: dict[str, pd.DataFrame], scores: dict,
                   official_test_csv: Path) -> dict:
    comparison = pd.read_csv(root / "per_image_comparison.csv")
    summary = load_json(root / "summary.json")
    test = pd.read_csv(official_test_csv)
    require(comparison.Id.tolist() == list(range(1, 10))
            and comparison.image.tolist() == test.image.tolist()
            and comparison.category.tolist() == test.category.tolist(),
            "Per-image comparison identity/order changed")
    pieces = {method: row_parts(frame) for method, frame in frames.items()}
    row_score = {method: sum(pieces[method].values()) for method in frames}
    for method, frame in frames.items():
        close(summary[method]["score_from_rows"], scores[method]["score"],
              f"Summary score {method}")
        for metric in METRICS:
            same_column(comparison, f"{method}_{metric}", frame[metric].to_numpy(dtype=float))
            close(summary[method]["means"][metric], float(frame[metric].mean()),
                  f"Summary mean {method}/{metric}")
        for term in TERMS:
            same_column(comparison, f"{method}_{term}", pieces[method][term])
            close(summary[method]["contributions"][term],
                  float(pieces[method][term].mean()),
                  f"Summary contribution {method}/{term}")
        same_column(comparison, f"{method}_row_score", row_score[method])
    for method, baseline, key in (
        (PREVIOUS, CONTROL, "vs_control"),
        (CANDIDATE, CONTROL, "vs_control"),
        (CANDIDATE, PREVIOUS, "vs_stage21"),
    ):
        details = summary[method][key]
        for metric in METRICS:
            delta = (frames[method][metric] - frames[baseline][metric]).to_numpy(dtype=float)
            same_column(comparison, f"{method}_delta_{metric}_vs_{baseline}", delta)
            close(details["metric_mean_deltas"][metric], float(delta.mean()),
                  f"Mean delta {method}/{baseline}/{metric}")
        for term in TERMS:
            delta = pieces[method][term] - pieces[baseline][term]
            same_column(comparison, f"{method}_delta_{term}_vs_{baseline}", delta)
            close(details["contribution_mean_deltas"][term], float(delta.mean()),
                  f"Contribution delta {method}/{baseline}/{term}")
        delta_score = row_score[method] - row_score[baseline]
        same_column(comparison, f"{method}_delta_row_score_vs_{baseline}", delta_score)
        close(details["score_delta"], float(delta_score.mean()), f"Score delta {method}/{baseline}")
        close(details["median_row_score_delta"], float(np.median(delta_score)),
              f"Median score delta {method}/{baseline}")
        require(details["images_improved"] == int((delta_score > 0).sum())
                and details["images_declined"] == int((delta_score < 0).sum())
                and details["improved_ids"] == [
                    int(value) for value in frame_ids(delta_score > 0)
                ], f"Per-image delta counts differ: {method}/{baseline}")
        close(details["id2_row_score_delta"], float(delta_score[1]),
              f"Id 2 row-score delta {method}/{baseline}")
        expected_share = (float(delta_score[1] / delta_score.sum())
                          if abs(float(delta_score.sum())) > 1e-12 else None)
        require((expected_share is None and details["id2_share_of_nine_row_delta_sum"] is None)
                or (expected_share is not None and
                    math.isclose(float(details["id2_share_of_nine_row_delta_sum"]),
                                 expected_share, rel_tol=0, abs_tol=1e-9)),
                f"Id 2 gain share differs: {method}/{baseline}")
    require(summary[CANDIDATE]["strictly_above_stage21"]
            is (scores[CANDIDATE]["score"] > SCORE_42[PREVIOUS]),
            "Stage 22 promotion condition differs")
    return {
        "stage22_score_delta_vs_control":
            scores[CANDIDATE]["score"] - scores[CONTROL]["score"],
        "stage22_score_delta_vs_stage21":
            scores[CANDIDATE]["score"] - scores[PREVIOUS]["score"],
        "stage22_images_improved_vs_control":
            summary[CANDIDATE]["vs_control"]["images_improved"],
        "stage22_images_improved_vs_stage21":
            summary[CANDIDATE]["vs_stage21"]["images_improved"],
    }


def frame_ids(mask: np.ndarray) -> list[int]:
    return [index + 1 for index, selected in enumerate(mask) if selected]


def verify_probe(root: Path, records: dict, test: pd.DataFrame) -> dict:
    probe = load_json(root / "probe.json")
    require(probe.get("status") == "PASS" and probe.get("all_checks_passed") is True,
            "8x8 probe did not pass")
    expected = {
        "weights": records["weights"]["sha256"],
        "official_generator": records["official_generator"]["sha256"],
        "official_metric": records["official_scorer"]["sha256"],
        "experiment": records["stage22_code/experiment.py"]["sha256"],
        "probe": records["stage22_code/probe.py"]["sha256"],
        "official_image": records[f"test_image/{test.iloc[0].image}"]["sha256"],
    }
    for name, digest in expected.items():
        require(probe["sources"][name]["sha256"] == digest, f"Probe source hash differs: {name}")
    require(probe["perturbation"]["baseline_changed"] is True
            and probe["original"]["input_sha256"] != probe["perturbed"]["input_sha256"],
            "Probe perturbed input/baseline identity failed")
    batch_size = int(probe["original"]["benchmark"]["inference_batch_size"])
    require(batch_size == int(probe["perturbed"]["benchmark"]["inference_batch_size"])
            and probe["environment"]["torch_num_threads"] == 8
            and 1 <= batch_size <= 32, "Probe thread/batch configuration differs")
    checked = 0
    for name in ("original", "perturbed"):
        case = probe[name]
        close(case["baseline_from_entire_chw"], case["input_global_min"],
              f"Full CHW black baseline {name}", tol=0)
        require(case["input_shape"] == [3, 224, 224]
                and case["input_dtype"] == "float32"
                and case["raw_attribution_shape"] == list(EXPECTED_SHAPE)
                and case["attribution_shape"] == list(EXPECTED_SHAPE)
                and case["raw_attribution_dtype"] == "float32"
                and case["attribution_dtype"] == "float32"
                and case["raw_attribution_finite"] is True
                and case["attribution_finite"] is True
                and case["all_tile_masks_audited"] == 2352
                and case["all_tiles_partition_complete_chw_once"] is True
                and case["selected_count_per_tile"] == 64,
                f"Full 2352-mask/raw/final probe audit failed: {name}")
        bench = case["benchmark"]
        stats = bench["implementation_stats"]
        require(bench["masked_inputs_expected"] == 2352
                and stats["masked_inputs_evaluated"] == 2352
                and stats["original_inputs_evaluated"] == 1
                and stats["computed_attributions"] == 1
                and stats["model_forward_calls"] == 1 + math.ceil(2352 / batch_size)
                and bench["wall_seconds"] > 0
                and bench["rss_peak_sampled_bytes"] > 0,
                f"Probe runtime/model counts failed: {name}")
        require(len(case["tiles"]) == 2, f"Expected first/last logit checks: {name}")
        for tile in case["tiles"]:
            require(tile["tile_index"] in (0, 2351)
                    and tile["selected_count"] == 64
                    and tile["selected_replacement_exact"] is True
                    and tile["unselected_bytes_unchanged"] is True
                    and tile["raw_tile_constant"] is True,
                    f"Independent mask/logit tile audit failed: {name}")
            close(tile["actual_raw_r_per_element"], tile["expected_r_per_element"],
                  f"Independent logit-difference attribution {name}", tol=2e-6)
            checked += 1
    return {"status": "PASS", "audited_masks": 4704,
            "independent_logit_checks": checked,
            "inference_batch_size": batch_size}


def verify_candidate_run(root: Path, run: dict, records: dict) -> dict:
    require(run.get("method_code") == "stage22-8x8-black-occlusion-signed-v1"
            and run.get("method_noise") == "none",
            "Stage 22 scored run method version/noise mismatch")
    cache = run["attribution_cache"]
    spec = cache["spec"]
    config = load_json(root / "cache" / CANDIDATE / "config.json")
    require(spec == config, "Stage 22 cache config differs from scored run")
    require(Path(cache["root"]).resolve() == (root / "cache" / CANDIDATE).resolve(),
            "Stage 22 scored cache path differs")
    require(spec["schema"] == 2 and spec["method"] == CANDIDATE
            and spec["method_version"] == run["method_code"]
            and spec["code_sha256"] == FIXED_HASHES["stage22_code/experiment.py"]
            and spec["weight_sha256"] == FIXED_HASHES["weights"]
            and re.fullmatch(r"[0-9a-f]{64}", spec["occlusion_core_sha256"]) is not None
            and spec["tile_size"] == 8 and spec["tile_count"] == 2352
            and spec["tile_area"] == 64
            and spec["noise"] == "none"
            and spec["inference_batch_size"] == 32
            and spec["torch_num_threads"] == 8
            and spec["top_fraction_first"] == 0.25
            and spec["top_fraction_last"] == 0.10
            and spec["stored_arrays"] == ["raw_attribution", "attribution"],
            "Stage 22 cache identity or method parameters mismatch")
    require(spec["torch"] == run["runtime"]["torch"]
            and spec["numpy"] == run["runtime"]["numpy"],
            "Cache and scored runtime version mismatch")
    occ = run["occlusion"]
    computed = int(occ["computed_attributions"])
    require(occ["tile_size"] == 8 and occ["tile_count_per_input"] == 2352
            and occ["inference_batch_size"] == 32
            and occ["torch_num_threads"] == 8
            and occ["torch_num_threads_actual"] == 8
            and computed == cache["misses"] == cache["writes"]
            and computed + cache["memory_hits"] + cache["disk_hits"] == 72
            and occ["masked_inputs_evaluated"] == 2352 * computed
            and occ["original_inputs_evaluated"] == computed
            and occ["model_forward_calls"] == 75 * computed
            and run["elapsed_seconds"] > 0 and occ["peak_rss_bytes"] > 0
            and cache["invalid_entries"] == 0,
            "Stage 22 scored callback/cache/model counts differ")
    return {"computed_attributions": computed,
            "masked_inputs_evaluated": occ["masked_inputs_evaluated"],
            "cache_hits": cache["memory_hits"] + cache["disk_hits"],
            "elapsed_seconds": run["elapsed_seconds"],
            "peak_rss_bytes": occ["peak_rss_bytes"]}


def reproduce_final(raw: np.ndarray) -> np.ndarray:
    """Independently execute the documented two rankings and reflect mean filter."""
    flat = raw.reshape(raw.shape[0], -1)
    first = np.zeros_like(flat)
    keep_first = int(np.ceil(flat.shape[1] * 0.25))
    for index, row in enumerate(flat):
        chosen = np.argsort(np.abs(row), kind="stable")[-keep_first:]
        first[index, chosen] = row[chosen]
    tensor = torch.from_numpy(np.ascontiguousarray(first.reshape(raw.shape)))
    averaged = F.avg_pool2d(F.pad(tensor, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
    smooth = averaged.numpy().reshape(raw.shape[0], -1)
    final = np.zeros_like(smooth)
    keep_last = int(np.ceil(smooth.shape[1] * 0.10))
    for index, row in enumerate(smooth):
        chosen = np.argsort(np.abs(row), kind="stable")[-keep_last:]
        final[index, chosen] = row[chosen]
    return np.ascontiguousarray(final.reshape(raw.shape), dtype=np.float32)


def verify_cache(root: Path, minimum_shards: int) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    cache = root / "cache" / CANDIDATE
    shards = sorted(cache.rglob("*.npz"))
    require(len(shards) >= minimum_shards and len(shards) >= 54,
            f"Only {len(shards)} Stage 22 cache shards")
    maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    total_bytes = 0
    for path in shards:
        key = path.stem
        require(re.fullmatch(r"[0-9a-f]{64}", key) is not None
                and path.parent == cache / key[:2],
                f"Invalid content-addressed cache path: {path}")
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"raw_attribution", "attribution", "metadata"},
                    f"Wrong cache arrays: {path}")
            raw = np.ascontiguousarray(archive["raw_attribution"])
            final = np.ascontiguousarray(archive["attribution"])
            meta = json.loads(str(archive["metadata"].item()))
        require(meta.get("key") == key
                and meta.get("method_version") == "stage22-8x8-black-occlusion-signed-v1"
                and meta.get("shape") == list(EXPECTED_SHAPE)
                and meta.get("dtype") == np.dtype(np.float32).str
                and raw.shape == EXPECTED_SHAPE and final.shape == EXPECTED_SHAPE
                and raw.dtype == np.float32 and final.dtype == np.float32
                and np.isfinite(raw).all() and np.isfinite(final).all()
                and meta.get("raw_attribution_sha256") == sha256_array(raw)
                and meta.get("attribution_sha256") == sha256_array(final),
                f"Cache raw/final metadata, shape, values or SHA-256 invalid: {path}")
        tile_view = raw[0].reshape(3, 28, 8, 28, 8)
        tile_first = tile_view[:, :, :1, :, :1]
        require(np.all(tile_view == tile_first),
                f"Raw attribution is not constant within 8x8 CHW tiles: {path}")
        reproduced = reproduce_final(raw)
        require(np.array_equal(reproduced.view(np.uint8), final.view(np.uint8)),
                f"Cached final map differs from independent postprocessing: {path}")
        maps[key] = raw, final
        total_bytes += path.stat().st_size
    return {"status": "PASS", "validated_shards": len(shards),
            "total_shard_bytes": total_bytes}, maps


def official_items(paths: dict[str, Path]) -> list[dict]:
    official = paths["official_test_csv"].parent.parent
    sys.path.insert(0, str(official / "vendor"))
    from kaggle.model.dataset.ship_image_datasets import ShipImageDataset
    from kaggle.model.dataset.transforms import val_transforms

    dataset = ShipImageDataset(str(paths["official_test_csv"]),
                               str(official / "data/test_images"), transform=val_transforms)
    test = pd.read_csv(paths["official_test_csv"])
    require(len(dataset) == 9, "Official dataset count differs")
    result = []
    for index in range(9):
        tensor, target = dataset[index]
        x = np.ascontiguousarray(tensor.detach().cpu().numpy()[None], dtype=np.float32)
        label = np.ascontiguousarray(np.array([int(target.item())], dtype=np.int64))
        filename = str(test.iloc[index].image)
        require(x.shape == EXPECTED_SHAPE
                and int(target.item()) + 1 == int(test.iloc[index].category),
                f"Official transformed image shape/target differs: Id {index + 1}")
        transformed = np.rint(np.clip(
            np.transpose(x[0], (1, 2, 0)) * STD + MEAN, 0, 1
        ) * 255).astype(np.uint8)
        result.append({"Id": index + 1, "filename": filename,
                       "target": int(target.item()), "x": x,
                       "label": label, "transformed": transformed})
    return result


def cache_key(x: np.ndarray, target: np.ndarray, spec: dict) -> str:
    identity = {
        "spec": spec, "input_shape": list(x.shape),
        "input_dtype": x.dtype.str, "input_sha256": sha256_array(x),
        "target_shape": list(target.shape), "target_dtype": target.dtype.str,
        "target_sha256": sha256_array(target),
    }
    data = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def png_pixels(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    require(path.is_file() and path.stat().st_size > 0, f"PNG missing or empty: {path}")
    with Image.open(path) as picture:
        require(picture.format == "PNG" and picture.size == expected_size,
                f"PNG format/size invalid: {path}")
        picture.load()
        return np.asarray(picture.convert("RGB"))


def safe_png_path(root: Path, value: str, method: str) -> Path:
    path = (root / value).resolve()
    require(path.is_relative_to((root / "heatmaps" / method).resolve())
            and path.suffix.lower() == ".png",
            f"Heatmap PNG path outside method folder: {value}")
    return path


def expected_overlay(rgb: np.ndarray, absolute: np.ndarray, scale: float) -> np.ndarray:
    strength = np.clip(absolute / max(scale, 1e-15), 0, 1) ** 0.6
    base = rgb.astype(np.float64) / 255
    color = np.stack((np.ones_like(strength), 0.6 * (1 - strength),
                      np.zeros_like(strength)), axis=-1)
    alpha = (0.7 * strength)[..., None]
    return np.rint(np.clip(base * (1 - alpha) + color * alpha, 0, 1) * 255).astype(np.uint8)


def expected_signed(signed: np.ndarray, scale: float) -> np.ndarray:
    value = np.clip(signed / max(scale, 1e-15), -1, 1)
    strength = np.abs(value) ** 0.6
    neutral = np.full((*value.shape, 3), 235.0)
    positive = np.stack((np.full_like(value, 212), np.full_like(value, 44),
                         np.full_like(value, 46)), axis=-1)
    negative = np.stack((np.full_like(value, 47), np.full_like(value, 95),
                         np.full_like(value, 211)), axis=-1)
    color = np.where((value >= 0)[..., None], positive, negative)
    return np.rint(np.clip(neutral * (1 - strength[..., None])
                           + color * strength[..., None], 0, 255)).astype(np.uint8)


def percentile_scale(arrays: list[np.ndarray]) -> float:
    positive = np.concatenate([x[x > 0].reshape(-1) for x in arrays])
    return float(np.quantile(positive, 0.99)) if positive.size else 1.0


def verify_heatmaps(root: Path, paths: dict[str, Path], records: dict,
                    cache_maps: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    manifest = load_json(root / "heatmaps/manifest.json")
    observations = load_json(root / "heatmaps/observations.json")
    require(manifest.get("status") == "NINE_ORIGINAL_INPUT_CACHE_MAPS_VERIFIED"
            and manifest.get("methods") == EXPECTED_METHODS
            and manifest.get("observations_complete") is True
            and manifest.get("stage22_raw_to_final_postprocess_verified") is True
            and manifest.get("observations_file") == "heatmaps/observations.json"
            and manifest.get("observations_sha256")
                == sha256_file(root / "heatmaps/observations.json")
            and manifest.get("official_generator_sha256") == FIXED_HASHES["official_generator"]
            and manifest.get("official_metrix_sha256") == FIXED_HASHES["official_scorer"]
            and manifest.get("weight_sha256_from_scored_runs") == FIXED_HASHES["weights"]
            and manifest.get("official_test_csv_sha256")
                == records["official_test_csv"]["sha256"],
            "Both-method heatmap manifest source/observations identity failed")
    require(manifest["program_sha256"] == {
        "stage21_16x16": FIXED_HASHES["stage21_experiment"],
        "stage22_8x8": FIXED_HASHES["stage22_code/experiment.py"],
    }, "Heatmap program identity differs")
    entries = manifest["entries"]
    require(len(entries) == 18, "Expected 18 original-input heatmap entries")
    items = official_items(paths)
    specs = {
        "stage21_16x16": load_json(paths["stage21_cache_config"]),
        "stage22_8x8": load_json(root / "cache" / CANDIDATE / "config.json"),
    }
    require(specs["stage21_16x16"] == load_json(paths["stage21_run"])["attribution_cache"]["spec"],
            "Stage 21 original cache spec differs from scored run")
    cache_roots = {
        "stage21_16x16": WORKSPACE / "output_mac_stage21/cache" / PREVIOUS,
        "stage22_8x8": root / "cache" / CANDIDATE,
    }
    for method in EXPECTED_METHODS:
        observed = observations.get(method)
        require(isinstance(observed, list) and len(observed) == 9
                and [row.get("Id") for row in observed] == list(range(1, 10)),
                f"Missing nine human observations: {method}")
        for index, row in enumerate(observed):
            require(row.get("filename") == items[index]["filename"]
                    and isinstance(row.get("visual_observation_zh"), str)
                    and row["visual_observation_zh"].strip()
                    and isinstance(row.get("off_ship_heat_zh"), str)
                    and row["off_ship_heat_zh"].strip(),
                    f"Incomplete observation for {method} Id {index + 1}")
    require(isinstance(observations.get("limitation_zh"), str)
            and observations["limitation_zh"].strip(),
            "Missing visual interpretation limitation")
    absolute_arrays, signed_arrays, raw_signed_arrays = [], [], []
    stored: list[tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for method in EXPECTED_METHODS:
        method_entries = [row for row in entries if row.get("method") == method]
        require([row.get("Id") for row in method_entries] == list(range(1, 10)),
                f"Heatmap Id order/count differs: {method}")
        for index, entry in enumerate(method_entries):
            item = items[index]
            x, label = item["x"], item["label"]
            key = cache_key(x, label, specs[method])
            cache_path = cache_roots[method] / key[:2] / f"{key}.npz"
            require(entry.get("filename") == item["filename"]
                    and entry.get("target_class_zero_based") == item["target"]
                    and entry.get("target_category_one_based") == item["target"] + 1
                    and entry.get("transformed_input_shape") == list(x.shape)
                    and entry.get("transformed_input_dtype") == x.dtype.str
                    and entry.get("transformed_input_sha256") == sha256_array(x)
                    and entry.get("target_shape") == list(label.shape)
                    and entry.get("target_dtype") == label.dtype.str
                    and entry.get("target_sha256") == sha256_array(label)
                    and entry.get("cache_key") == key
                    and entry.get("cache_file")
                        == str(cache_path.relative_to(WORKSPACE))
                    and entry.get("cache_file_sha256") == sha256_file(cache_path)
                    and entry.get("source_jpeg_sha256")
                        == records[f"test_image/{item['filename']}"]["sha256"],
                    f"Heatmap exact image/target/cache identity failed: {method} Id {index + 1}")
            preserved = root / "heatmaps/official_sources" / item["filename"]
            require(entry.get("preserved_original_jpeg")
                        == str(preserved.relative_to(root))
                    and sha256_file(preserved)
                        == records[f"test_image/{item['filename']}"]["sha256"],
                    f"Heatmap preserved source JPEG differs: Id {index + 1}")
            if method == "stage22_8x8":
                require(key in cache_maps, f"Stage 22 original-input cache missing: Id {index + 1}")
                raw, final = cache_maps[key]
                require(entry.get("raw_attribution_sha256") == sha256_array(raw),
                        f"Stage 22 heatmap raw map hash differs: Id {index + 1}")
            else:
                with np.load(cache_path, allow_pickle=False) as archive:
                    require(set(archive.files) == {"attribution", "metadata"},
                            f"Stage 21 cache fields differ: Id {index + 1}")
                    final = np.ascontiguousarray(archive["attribution"])
                    meta = json.loads(str(archive["metadata"].item()))
                require(meta.get("key") == key and meta.get("attribution_sha256")
                        == sha256_array(final) and final.shape == EXPECTED_SHAPE
                        and final.dtype == np.float32 and np.isfinite(final).all(),
                        f"Stage 21 original cache corrupt: Id {index + 1}")
                raw = None
                require(entry.get("raw_attribution_sha256") is None,
                        f"Stage 21 raw map falsely claimed: Id {index + 1}")
            require(entry.get("final_attribution_sha256") == sha256_array(final),
                    f"Heatmap final attribution hash differs: {method} Id {index + 1}")
            absolute = np.abs(final[0]).sum(axis=0, dtype=np.float64)
            signed = final[0].sum(axis=0, dtype=np.float64)
            close(entry["absolute_heat_total"], float(absolute.sum()),
                  f"Heatmap absolute total {method} Id {index + 1}")
            require(entry["positive_signed_pixel_count"] == int(np.count_nonzero(signed > 0))
                    and entry["negative_signed_pixel_count"] == int(np.count_nonzero(signed < 0)),
                    f"Heatmap signed counts differ: {method} Id {index + 1}")
            require(entry.get("observation") == observations[method][index],
                    f"Manifest and human observation differ: {method} Id {index + 1}")
            absolute_arrays.append(absolute)
            signed_arrays.append(np.abs(signed))
            raw_signed = None
            if raw is not None:
                raw_signed = raw[0].sum(axis=0, dtype=np.float64)
                raw_signed_arrays.append(np.abs(raw_signed))
            stored.append((entry, item["transformed"], absolute, signed, raw_signed))
    absolute_scale = percentile_scale(absolute_arrays)
    signed_scale = percentile_scale(signed_arrays)
    raw_signed_scale = percentile_scale(raw_signed_arrays)
    close(manifest["absolute_scale"], absolute_scale, "Shared absolute heatmap scale")
    close(manifest["signed_scale"], signed_scale, "Shared signed heatmap scale")
    close(manifest["raw_signed_scale"], raw_signed_scale, "Shared raw signed heatmap scale")
    for entry, rgb, absolute, signed, raw_signed in stored:
        method = entry["method"]
        source_path = safe_png_path(root, entry["model_input_png"], method)
        overlay_path = safe_png_path(root, entry["overlay_png"], method)
        signed_path = safe_png_path(root, entry["signed_png"], method)
        require(np.array_equal(png_pixels(source_path, (224, 224)), rgb),
                f"Model-input PNG pixels differ: {method} Id {entry['Id']}")
        require(np.array_equal(png_pixels(overlay_path, (224, 224)),
                               expected_overlay(rgb, absolute, absolute_scale)),
                f"Overlay PNG pixels differ: {method} Id {entry['Id']}")
        require(np.array_equal(png_pixels(signed_path, (224, 224)),
                               expected_signed(signed, signed_scale)),
                f"Signed PNG pixels differ: {method} Id {entry['Id']}")
        triptych = safe_png_path(root, entry["triptych_png"], method)
        contact = safe_png_path(root, entry["contact_sheet_png"], method)
        png_pixels(triptych, (672, 273))
        png_pixels(contact, (2016, 819))
        if raw_signed is not None:
            raw_path = safe_png_path(root, entry["raw_signed_png"], method)
            require(np.array_equal(png_pixels(raw_path, (224, 224)),
                                   expected_signed(raw_signed, raw_signed_scale)),
                    f"Raw signed PNG pixels differ: Id {entry['Id']}")
        else:
            require(entry.get("raw_signed_png") is None,
                    f"Stage 21 raw PNG falsely claimed: Id {entry['Id']}")
    return {"status": "PASS", "methods": EXPECTED_METHODS,
            "maps_verified": len(entries), "human_observations": 18,
            "shared_absolute_scale": absolute_scale,
            "shared_signed_scale": signed_scale}


def verify_branch(root: Path, candidate_score: float, scorer, solution: pd.DataFrame,
                  records: dict, candidate_experiment: Path,
                  control_experiment: Path) -> dict:
    decision_path = root / "branch_decision.json"
    above = candidate_score > SCORE_42[PREVIOUS]
    if not above:
        if decision_path.exists():
            decision = load_json(decision_path)
            require(decision.get("hotspots_roughly_ship") is not True,
                    "Branch decision claims promotion despite score gate failure")
        return {"score_strictly_above_stage21": False,
                "paired_repeats_required": False}
    require(decision_path.is_file(), "Missing visual branch decision after score gate")
    decision = load_json(decision_path)
    plausible = decision.get("hotspots_roughly_ship")
    require(isinstance(plausible, bool), "Branch decision lacks boolean hotspot judgement")
    if not plausible:
        return {"score_strictly_above_stage21": True,
                "hotspots_roughly_ship": False,
                "paired_repeats_required": False}
    paired: dict[str, dict] = {}
    for seed in (314159, 271828):
        seed_key = str(seed)
        paired[seed_key] = {}
        for method, experiment in ((CONTROL, control_experiment),
                                   (CANDIDATE, candidate_experiment)):
            stem = f"{method}_seed{seed}"
            base = root / "paired_runs" / stem
            csv_path = base.with_suffix(".csv")
            score_path = root / "paired_runs" / f"{stem}.score.json"
            run_path = root / "paired_runs" / f"{stem}.run.json"
            log_path = root / "paired_runs" / f"{stem}.log"
            require(log_path.is_file() and log_path.stat().st_size > 0,
                    f"Paired run raw log missing: {log_path}")
            result, frame, run = verify_scored_run(
                csv_path, score_path, run_path, method, seed,
                scorer, solution, records, experiment,
            )
            if method == CANDIDATE:
                verify_candidate_run(root, run, records)
            paired[seed_key][method] = result
            paired[seed_key][f"{method}_frame"] = frame
        c = paired[seed_key][CANDIDATE]
        b = paired[seed_key][CONTROL]
        cframe = paired[seed_key].pop(f"{CANDIDATE}_frame")
        bframe = paired[seed_key].pop(f"{CONTROL}_frame")
        delta = sum(row_parts(cframe).values()) - sum(row_parts(bframe).values())
        paired[seed_key]["delta_candidate_minus_control"] = float(c["score"] - b["score"])
        paired[seed_key]["per_image_deltas"] = [float(value) for value in delta]
        paired[seed_key]["images_improved"] = int((delta > 0).sum())
    return {"score_strictly_above_stage21": True,
            "hotspots_roughly_ship": True,
            "paired_repeats_required": True,
            "paired": paired}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    environment, paths = environment_sources(root)
    records = environment["files"]
    scorer = official_score_function(paths["official_scorer"])
    solution = pd.read_csv(paths["official_solution"])
    tested = pd.read_csv(paths["official_test_csv"])
    sources = {
        CONTROL: (
            paths["prior_control_csv"], WORKSPACE / "output_mac_next/runs/control.score.json",
            paths["prior_control_run"], paths["prior_control_experiment"],
        ),
        PREVIOUS: (
            paths["stage21_csv"], paths["stage21_score"], paths["stage21_run"],
            paths["stage21_experiment"],
        ),
        CANDIDATE: (
            root / "runs" / f"{CANDIDATE}.csv",
            root / "runs" / f"{CANDIDATE}.score.json",
            root / "runs" / f"{CANDIDATE}.run.json",
            paths["stage22_code/experiment.py"],
        ),
    }
    verified, frames, runs = {}, {}, {}
    for method, (csv_path, score_path, run_path, experiment_path) in sources.items():
        verified[method], frames[method], runs[method] = verify_scored_run(
            csv_path, score_path, run_path, method, 42, scorer, solution,
            records, experiment_path,
        )
    official_log = root / "runs" / f"{CANDIDATE}.log"
    require(official_log.is_file() and official_log.stat().st_size > 0,
            "Stage 22 official nine-image raw log missing")
    for method, expected in SCORE_42.items():
        close(verified[method]["score"], expected, f"Read-only {method} seed 42 score")
    comparison = verify_summary(root, frames, verified, paths["official_test_csv"])
    probe = verify_probe(root, records, tested)
    candidate_run = verify_candidate_run(root, runs[CANDIDATE], records)
    cache, maps = verify_cache(root, int(
        runs[CANDIDATE]["attribution_cache"]["writes"]
        + runs[CANDIDATE]["attribution_cache"]["disk_hits"]
    ))
    heatmaps = verify_heatmaps(root, paths, records, maps)
    branch = verify_branch(root, verified[CANDIDATE]["score"], scorer, solution, records,
                           paths["stage22_code/experiment.py"],
                           paths["prior_control_experiment"])
    pending = [str(path) for path in root.rglob("*.pending*")]
    require(not pending, f"Pending output files remain: {pending}")
    result = {
        "status": "PASS",
        "scope": "nine organizer-supplied images; unseen platform outcome unknown",
        "environment_sources_verified": len(records),
        "official_scores": verified,
        "per_image_comparison": comparison,
        "probe": probe,
        "stage22_scored_run": candidate_run,
        "resumable_raw_and_final_cache": cache,
        "both_method_heatmaps": heatmaps,
        "conditional_branch": branch,
    }
    atomic_json(root / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
