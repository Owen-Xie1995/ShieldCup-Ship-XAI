"""Reconstruct organizer-image heatmaps from verified, exact-input cache entries.

This is a post-hoc read-only consumer of the Stage 21 and Stage 22 caches.  It
never computes a missing attribution, and no annotation enters official scoring.
Run Stage 21 first; after the 8x8 run, run ``--method both`` for a common scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
STAGE17 = Path("/Users/xiefujun/Downloads/stage17")
EXPECTED_GENERATOR = "3f0c8830e590f6ad46934c034a5bf75f2d2b1d89baf2dba6683cdc1d9dc61df1"
EXPECTED_METRIX = "5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda"
EXPECTED_WEIGHT = "eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473"
EXPECTED_STAGE21_CODE = "5264364073961e5c3f8de1e1982fe2856de4eeecca40aca4ca5ca459784360f5"
EXPECTED_TEST_CSV = "292bf8cf64b4db3a64def4a27ec02f082bd7b222e8ae939e2d620d8409a8e83e"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def rgba_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
              fill: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=fill, font=font)


@dataclass
class Item:
    id: int
    filename: str
    target: int
    source: Path
    tensor: np.ndarray
    target_array: np.ndarray
    transformed: np.ndarray


@dataclass
class Map:
    item: Item
    method: str
    attr: np.ndarray
    raw: np.ndarray | None
    cache_key: str
    cache_path: Path
    cache_sha256: str
    attr_sha256: str
    raw_sha256: str | None


def official_items(stage17: Path) -> list[Item]:
    official = stage17 / "official"
    generator = official / "vendor/kaggle/submmit/submmit.py"
    metrix = official / "script/metrix.py"
    if sha256_file(generator) != EXPECTED_GENERATOR or sha256_file(metrix) != EXPECTED_METRIX:
        raise RuntimeError("Organizer generator or metrix.py SHA-256 changed")
    if sha256_file(official / "data/test.csv") != EXPECTED_TEST_CSV:
        raise RuntimeError("Organizer nine-image order or targets changed")
    historical_files = json.loads((WORKSPACE / "output_mac_stage21/environment.json")
                                  .read_text(encoding="utf-8"))["files"]
    sys.path.insert(0, str(official / "vendor"))
    from kaggle.model.dataset.ship_image_datasets import ShipImageDataset
    from kaggle.model.dataset.transforms import val_transforms

    dataset = ShipImageDataset(str(official / "data/test.csv"),
                               str(official / "data/test_images"), transform=val_transforms)
    if len(dataset) != 9:
        raise RuntimeError(f"Expected nine supplied images, got {len(dataset)}")
    result: list[Item] = []
    for index in range(len(dataset)):
        tensor, target = dataset[index]
        image_name = str(dataset.annotations.iloc[index]["image"])
        source = official / "data/test_images" / image_name
        if not source.is_file():
            raise FileNotFoundError(source)
        historical_sha = historical_files[f"test_image/{image_name}"]["sha256"]
        if sha256_file(source) != historical_sha:
            raise RuntimeError(f"Supplied image bytes changed for Id {index + 1}")
        x = np.ascontiguousarray(tensor.detach().cpu().numpy()[None], dtype=np.float32)
        target_array = np.ascontiguousarray(np.array([int(target.item())], dtype=np.int64))
        if x.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Wrong official image shape for Id {index + 1}: {x.shape}")
        image = np.clip(np.transpose(x[0], (1, 2, 0)) * STD + MEAN, 0, 1)
        transformed = np.rint(image * 255).astype(np.uint8)
        result.append(Item(index + 1, image_name, int(target.item()), source,
                           x, target_array, transformed))
    return result


def load_method_spec(method: str) -> tuple[Path, Path, dict, dict]:
    if method == "stage21_16x16":
        stage = WORKSPACE / "output_mac_stage21"
        code_hash = EXPECTED_STAGE21_CODE
        run_name = "black_occlusion16_signed"
    elif method == "stage22_8x8":
        stage = ROOT
        code_hash = None
        run_name = "black_occlusion8_signed"
    else:
        raise ValueError(method)
    code_path = stage / "experiment.py"
    actual_code_hash = sha256_file(code_path)
    if code_hash is not None and actual_code_hash != code_hash:
        raise RuntimeError(f"Stage 21 code hash differs: {actual_code_hash}")
    run = json.loads((stage / "runs" / f"{run_name}.run.json").read_text(encoding="utf-8"))
    if run.get("evaluation_seed") != 42 or run.get("method") != run_name:
        raise RuntimeError(f"Scored run does not match the expected method and seed: {method}")
    if Path(run["paths"]["stage17"]).resolve() != STAGE17.resolve():
        raise RuntimeError(f"Scored run used another organizer directory: {method}")
    if run.get("official_weight_sha256") != EXPECTED_WEIGHT:
        raise RuntimeError("Scored run used different official weights")
    run_hashes = run.get("code_sha256", {})
    if (run_hashes.get("experiment") != actual_code_hash
            or run_hashes.get("official_generator") != EXPECTED_GENERATOR
            or run_hashes.get("official_scorer") != EXPECTED_METRIX):
        raise RuntimeError(f"Scored run program or source hash mismatch: {method}")
    cache = stage / "cache" / run_name
    spec = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    if spec != run["attribution_cache"]["spec"]:
        raise RuntimeError(f"Cache specification differs from scored run: {method}")
    if (spec.get("method") != run_name or spec.get("weight_sha256") != EXPECTED_WEIGHT
            or spec.get("code_sha256") != actual_code_hash):
        raise RuntimeError(f"Cache specification hash mismatch: {method}")
    if Path(run["attribution_cache"]["root"]).resolve() != cache.resolve():
        raise RuntimeError(f"Scored cache path mismatch: {method}")
    return cache, code_path, spec, run


def key_for(item: Item, spec: dict) -> str:
    x = item.tensor
    target = item.target_array
    identity = {
        "spec": spec,
        "input_shape": list(x.shape),
        "input_dtype": x.dtype.str,
        "input_sha256": sha256_array(x),
        "target_shape": list(target.shape),
        "target_dtype": target.dtype.str,
        "target_sha256": sha256_array(target),
    }
    return hashlib.sha256(json_bytes(identity)).hexdigest()


def verify_raw_to_final(raw: np.ndarray, final: np.ndarray) -> None:
    """Independently check the prescribed CHW sparsify/reflect-mean/sparsify path."""
    import torch
    import torch.nn.functional as F

    def top_signed(value: np.ndarray, fraction: float) -> np.ndarray:
        flat = value.reshape(value.shape[0], -1)
        kept = np.zeros_like(flat)
        count = int(np.ceil(flat.shape[1] * fraction))
        for i, row in enumerate(flat):
            indices = np.argsort(np.abs(row), kind="stable")[-count:]
            kept[i, indices] = row[indices]
        return kept.reshape(value.shape)

    first = top_signed(raw, 0.25)
    tensor = torch.from_numpy(np.ascontiguousarray(first))
    smooth = F.avg_pool2d(F.pad(tensor, (2, 2, 2, 2), mode="reflect"), 5,
                          stride=1).numpy()
    reproduced = np.ascontiguousarray(top_signed(smooth, 0.10), dtype=np.float32)
    if not np.array_equal(reproduced.view(np.uint8), final.view(np.uint8)):
        raise RuntimeError("Cached final map is not the exact prescribed postprocessing of raw map")


def read_map(item: Item, method: str, cache: Path, spec: dict) -> Map:
    key = key_for(item, spec)
    path = cache / key[:2] / f"{key}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Exact original-input cache entry missing for Id {item.id}: {path}")
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        if method == "stage21_16x16":
            if fields != {"attribution", "metadata"}:
                raise RuntimeError(f"Unexpected Stage 21 cache fields for Id {item.id}: {fields}")
            raw = None
        else:
            if fields != {"attribution", "raw_attribution", "metadata"}:
                raise RuntimeError(f"Unexpected Stage 22 cache fields for Id {item.id}: {fields}")
            raw = np.ascontiguousarray(archive["raw_attribution"])
        attr = np.ascontiguousarray(archive["attribution"])
        meta = json.loads(str(archive["metadata"].item()))
    if (meta.get("key") != key or tuple(meta.get("shape", [])) != item.tensor.shape
            or meta.get("dtype") != np.dtype("float32").str
            or attr.shape != item.tensor.shape or attr.dtype != np.float32
            or not np.isfinite(attr).all()
            or meta.get("attribution_sha256") != sha256_array(attr)):
        raise RuntimeError(f"Invalid cache identity or final attribution for Id {item.id}: {path}")
    raw_sha = None
    if raw is not None:
        raw_sha = sha256_array(raw)
        if raw.shape != item.tensor.shape or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise RuntimeError(f"Invalid raw attribution for Id {item.id}: {path}")
        stored_raw_sha = meta.get("raw_attribution_sha256", meta.get("raw_sha256"))
        if stored_raw_sha != raw_sha or meta.get("method_version") != spec.get("method_version"):
            raise RuntimeError(f"Raw attribution checksum mismatch for Id {item.id}: {path}")
        verify_raw_to_final(raw, attr)
    return Map(item, method, attr, raw, key, path, sha256_file(path),
               sha256_array(attr), raw_sha)


def heat_values(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chw = value[0]
    absolute = np.abs(chw).sum(axis=0, dtype=np.float64)
    signed = chw.sum(axis=0, dtype=np.float64)
    return absolute, signed


def common_scale(values: list[np.ndarray]) -> float:
    positive = np.concatenate([x[x > 0].reshape(-1) for x in values])
    return float(np.quantile(positive, 0.99)) if positive.size else 1.0


def overlay(transformed: np.ndarray, heat: np.ndarray, scale: float) -> Image.Image:
    strength = np.clip(heat / max(scale, 1e-15), 0, 1) ** 0.6
    base = transformed.astype(np.float64) / 255
    # Yellow/red means stronger absolute attribution; the RGB source stays visible.
    color = np.stack((np.ones_like(strength), 0.6 * (1 - strength),
                      np.zeros_like(strength)), axis=-1)
    alpha = (0.7 * strength)[..., None]
    rgb = np.clip(base * (1 - alpha) + color * alpha, 0, 1)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8), "RGB")


def signed_image(signed: np.ndarray, scale: float) -> Image.Image:
    # Zero is neutral gray, red positive logit support, blue negative support.
    value = np.clip(signed / max(scale, 1e-15), -1, 1)
    strength = np.abs(value) ** 0.6
    neutral = np.full((*value.shape, 3), 235.0)
    positive = np.stack((np.full_like(value, 212), np.full_like(value, 44),
                         np.full_like(value, 46)), axis=-1)
    negative = np.stack((np.full_like(value, 47), np.full_like(value, 95),
                         np.full_like(value, 211)), axis=-1)
    color = np.where((value >= 0)[..., None], positive, negative)
    rgb = neutral * (1 - strength[..., None]) + color * strength[..., None]
    return Image.fromarray(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), "RGB")


def font() -> ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/System/Library/Fonts/Helvetica.ttc"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, 17)
    return ImageFont.load_default()


def make_triptych(source: Image.Image, over: Image.Image, signed: Image.Image,
                  label: str, image_font: ImageFont.ImageFont) -> Image.Image:
    w, h = source.size
    frame = Image.new("RGB", (3 * w, h + 49), "white")
    for j, picture in enumerate((source, over, signed)):
        frame.paste(picture, (j * w, 49))
    draw = ImageDraw.Draw(frame)
    rgba_text(draw, (7, 4), label, (20, 20, 20), image_font)
    for j, name in enumerate(("official transformed", "absolute overlay", "signed +red/-blue")):
        rgba_text(draw, (j * w + 7, 27), name, (70, 70, 70), ImageFont.load_default())
    return frame


def save_contact(triptychs: list[Image.Image], path: Path) -> None:
    if len(triptychs) != 9:
        raise RuntimeError("A contact sheet requires exactly nine images")
    w, h = triptychs[0].size
    sheet = Image.new("RGB", (w * 3, h * 3), "#eeeeee")
    for i, triptych in enumerate(triptychs):
        sheet.paste(triptych, ((i % 3) * w, (i // 3) * h))
    sheet.save(path)


def save_comparison(loaded: dict[str, list[Map]], absolute_scale: float,
                    signed_scale: float) -> list[str]:
    """Put matched Id images and both methods on the same display scales."""
    first = loaded["stage21_16x16"]
    second = loaded["stage22_8x8"]
    destination = ROOT / "heatmaps" / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    panels = []
    paths = []
    for old, new in zip(first, second, strict=True):
        if (old.item.id != new.item.id or old.item.filename != new.item.filename
                or sha256_array(old.item.tensor) != sha256_array(new.item.tensor)
                or sha256_array(old.item.target_array) != sha256_array(new.item.target_array)):
            raise RuntimeError("The two methods do not correspond to the same Id/input/target")
        item = old.item
        old_absolute, old_signed = heat_values(old.attr)
        new_absolute, new_signed = heat_values(new.attr)
        pictures = [
            Image.fromarray(item.transformed, "RGB"),
            overlay(item.transformed, old_absolute, absolute_scale),
            overlay(item.transformed, new_absolute, absolute_scale),
            signed_image(old_signed, signed_scale),
            signed_image(new_signed, signed_scale),
        ]
        w, h = pictures[0].size
        panel = Image.new("RGB", (5 * w, h + 49), "white")
        for j, picture in enumerate(pictures):
            panel.paste(picture, (j * w, 49))
        draw = ImageDraw.Draw(panel)
        title = f"Id {item.id} · {item.filename} · target category {item.target + 1} (index {item.target})"
        rgba_text(draw, (7, 4), title, (20, 20, 20), font())
        for j, name in enumerate(("official input", "16x16 overlay", "8x8 overlay",
                                  "16x16 signed", "8x8 signed")):
            rgba_text(draw, (j * w + 7, 27), name, (70, 70, 70), ImageFont.load_default())
        path = destination / f"id_{item.id:02d}_{Path(item.filename).stem}_comparison.png"
        panel.save(path)
        panels.append(panel)
        paths.append(str(path.relative_to(ROOT)))
    save_contact(panels, destination / "contact_sheet.png")
    return paths


def generate(methods: list[str], stage17: Path, include_raw: bool) -> dict:
    items = official_items(stage17)
    observation_path = ROOT / "heatmaps" / "observations.json"
    observation_document = json.loads(observation_path.read_text(encoding="utf-8"))
    observation_lookup = {(method, int(record["Id"])): record
                          for method in methods
                          for record in observation_document.get(method, [])}
    loaded: dict[str, list[Map]] = {}
    run_hashes: dict[str, str] = {}
    for method in methods:
        cache, code, spec, run = load_method_spec(method)
        maps = [read_map(item, method, cache, spec) for item in items]
        loaded[method] = maps
        run_hashes[method] = sha256_file(code)
    all_maps = [m for maps in loaded.values() for m in maps]
    absolute_scale = common_scale([heat_values(m.attr)[0] for m in all_maps])
    signed_scale = common_scale([np.abs(heat_values(m.attr)[1]) for m in all_maps])
    raw_scale = common_scale([np.abs(heat_values(m.raw)[1]) for m in all_maps
                              if m.raw is not None]) if include_raw else None
    output: dict = {
        "status": "NINE_ORIGINAL_INPUT_CACHE_MAPS_VERIFIED",
        "scope": "supplied nine organizer images; visual inspection only",
        "official_generator_sha256": EXPECTED_GENERATOR,
        "official_metrix_sha256": EXPECTED_METRIX,
        "weight_sha256_from_scored_runs": EXPECTED_WEIGHT,
        "program_sha256": run_hashes,
        "official_test_csv_sha256": EXPECTED_TEST_CSV,
        "observations_file": str(observation_path.relative_to(ROOT)),
        "observations_sha256": sha256_file(observation_path),
        "methods": methods,
        "scale_rule": "One shared 99th percentile of all positive per-pixel values across requested methods and images; linear clipping followed by power 0.6 for display. Absolute heat is channelwise sum(abs(final attribution)); signed heat is channelwise sum(final attribution).",
        "stage22_raw_to_final_postprocess_verified": "stage22_8x8" in methods,
        "stage22_raw_to_final_postprocess_check": (
            "Byte-equal CHW top-25% abs-rank signed, reflect-padded 5x5 mean, top-10% abs-rank signed for every original-image cache entry"
            if "stage22_8x8" in methods else None),
        "absolute_scale": absolute_scale,
        "signed_scale": signed_scale,
        "raw_signed_scale": raw_scale,
        "entries": [],
    }
    source_dir = ROOT / "heatmaps" / "official_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        preserved = source_dir / item.filename
        if preserved.exists() and sha256_file(preserved) != sha256_file(item.source):
            raise RuntimeError(f"Preserved original source differs for Id {item.id}")
        if not preserved.exists():
            shutil.copyfile(item.source, preserved)
    for method, maps in loaded.items():
        destination = ROOT / "heatmaps" / method
        destination.mkdir(parents=True, exist_ok=True)
        triptychs = []
        for m in maps:
            item = m.item
            absolute, signed = heat_values(m.attr)
            source = Image.fromarray(item.transformed, "RGB")
            over = overlay(item.transformed, absolute, absolute_scale)
            signed_picture = signed_image(signed, signed_scale)
            stem = f"id_{item.id:02d}_{Path(item.filename).stem}"
            source_path = destination / f"{stem}_model_input.png"
            over_path = destination / f"{stem}_overlay.png"
            signed_path = destination / f"{stem}_signed.png"
            source.save(source_path)
            over.save(over_path)
            signed_picture.save(signed_path)
            label = f"Id {item.id} · {item.filename} · target category {item.target + 1} (index {item.target})"
            triptych = make_triptych(source, over, signed_picture, label, font())
            triptych.save(destination / f"{stem}_triptych.png")
            triptychs.append(triptych)
            raw_path = None
            if include_raw and m.raw is not None:
                raw_signed = heat_values(m.raw)[1]
                raw_image = signed_image(raw_signed, raw_scale or 1.0)
                raw_path = destination / f"{stem}_raw_signed.png"
                raw_image.save(raw_path)
            entry = {
                "method": method, "Id": item.id, "filename": item.filename,
                "target_class_zero_based": item.target,
                "target_category_one_based": item.target + 1,
                "source_jpeg_sha256": sha256_file(item.source),
                "preserved_original_jpeg": str((source_dir / item.filename).relative_to(ROOT)),
                "transformed_input_shape": list(item.tensor.shape),
                "transformed_input_dtype": item.tensor.dtype.str,
                "transformed_input_sha256": sha256_array(item.tensor),
                "target_shape": list(item.target_array.shape),
                "target_dtype": item.target_array.dtype.str,
                "target_sha256": sha256_array(item.target_array),
                "cache_key": m.cache_key,
                "cache_file": str(m.cache_path.relative_to(WORKSPACE)),
                "cache_file_sha256": m.cache_sha256,
                "final_attribution_sha256": m.attr_sha256,
                "raw_attribution_sha256": m.raw_sha256,
                "absolute_heat_total": float(absolute.sum()),
                "positive_signed_pixel_count": int(np.count_nonzero(signed > 0)),
                "negative_signed_pixel_count": int(np.count_nonzero(signed < 0)),
                "model_input_png": str(source_path.relative_to(ROOT)),
                "overlay_png": str(over_path.relative_to(ROOT)),
                "signed_png": str(signed_path.relative_to(ROOT)),
                "raw_signed_png": str(raw_path.relative_to(ROOT)) if raw_path else None,
                "triptych_png": str((destination / f"{stem}_triptych.png").relative_to(ROOT)),
                "contact_sheet_png": str((destination / "contact_sheet.png").relative_to(ROOT)),
                "observation": observation_lookup.get((method, item.id)),
            }
            output["entries"].append(entry)
        save_contact(triptychs, destination / "contact_sheet.png")
    output["observations_complete"] = all(record["observation"] is not None
                                          for record in output["entries"])
    if set(methods) == {"stage21_16x16", "stage22_8x8"}:
        output["comparison_panels"] = save_comparison(loaded, absolute_scale,
                                                       signed_scale)
        output["comparison_contact_sheet"] = "heatmaps/comparison/contact_sheet.png"
    path = ROOT / "heatmaps" / "manifest.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage17-dir", type=Path, default=STAGE17)
    parser.add_argument("--method", choices=("stage21_16x16", "stage22_8x8", "both"),
                        default="stage21_16x16")
    parser.add_argument("--include-raw", action="store_true",
                        help="Also render Stage 22 signed raw maps, when present")
    args = parser.parse_args()
    methods = (["stage21_16x16", "stage22_8x8"] if args.method == "both"
               else [args.method])
    result = generate(methods, args.stage17_dir.expanduser().resolve(), args.include_raw)
    print(json.dumps({"status": result["status"], "methods": result["methods"],
                      "images_per_method": 9, "absolute_scale": result["absolute_scale"],
                      "signed_scale": result["signed_scale"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
