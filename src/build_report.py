"""Build the final Stage 22 Traditional Chinese report from checked artifacts.

Run after the official nine-image score, both-method heatmaps, observations,
and branch decision are complete. This program reads scored files and writes
only report.md; it never edits an official CSV or attribution cache. Re-run
record_environment.py and verify_outputs.py after writing the report because
the environment inventory records the report's SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
CONTROL = "control"
PREVIOUS = "black_occlusion16_signed"
CANDIDATE = "black_occlusion8_signed"
METHODS = (CONTROL, PREVIOUS, CANDIDATE)
METRICS = ("Faithfulness", "Max_Sensitivity", "Sparsity")
TERMS = ("faithfulness_contribution", "sensitivity_contribution", "sparsity_contribution")
SEEDS = (314159, 271828)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}: {value}")
    return result


def close(actual: object, expected: object, label: str, tolerance: float = 1e-8) -> None:
    if abs(number(actual, label) - number(expected, label)) > tolerance:
        raise ValueError(f"{label}: {actual} != {expected}")


def fmt(value: object, places: int = 6, signed: bool = False) -> str:
    return format(number(value, "report value"), f"{'+' if signed else ''}.{places}f")


def csv_rows(path: Path, required: set[str], length: int) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing CSV columns in {path}: {required}")
        rows = list(reader)
    if len(rows) != length:
        raise ValueError(f"Expected {length} rows in {path}, found {len(rows)}")
    return rows


def scores_and_rows(output: Path, control_root: Path, stage21_root: Path):
    summary = read_json(output / "summary.json")
    metric_columns = {f"{method}_{metric}" for method in METHODS for metric in METRICS}
    score_columns = {f"{method}_row_score" for method in METHODS}
    delta_columns = {
        f"{CANDIDATE}_delta_{metric}_vs_{baseline}"
        for baseline in (CONTROL, PREVIOUS) for metric in (*METRICS, "row_score")
    }
    rows = csv_rows(output / "per_image_comparison.csv",
                    {"Id", "image", "category"} | metric_columns | score_columns | delta_columns,
                    9)
    for index, row in enumerate(rows, 1):
        if int(row["Id"]) != index or not row["image"] or not row["category"]:
            raise ValueError(f"Invalid Id/image/category at comparison row {index}")
        for method in METHODS:
            f, m, s = (number(row[f"{method}_{metric}"], f"{method}/{metric}/Id{index}")
                       for metric in METRICS)
            expected = 20 * (1 + f) + 30 * max(1 - m / 2, 0) + 30 * s
            close(row[f"{method}_row_score"], expected, f"{method} Id{index} formula")
        for baseline in (CONTROL, PREVIOUS):
            for metric in (*METRICS, "row_score"):
                close(row[f"{CANDIDATE}_delta_{metric}_vs_{baseline}"],
                      number(row[f"{CANDIDATE}_{metric}"], "candidate")
                      - number(row[f"{baseline}_{metric}"], "baseline"),
                      f"Id{index} delta {metric} vs {baseline}")
    roots = {CONTROL: control_root, PREVIOUS: stage21_root, CANDIDATE: output}
    stems = {CONTROL: "control", PREVIOUS: PREVIOUS, CANDIDATE: CANDIDATE}
    scored = {}
    for method in METHODS:
        if method not in summary:
            raise ValueError(f"Missing summary method: {method}")
        stem = stems[method]
        run_dir = roots[method] / "runs"
        score = read_json(run_dir / f"{stem}.score.json")
        run = read_json(run_dir / f"{stem}.run.json")
        csv_path = run_dir / f"{stem}.csv"
        digest = sha256(csv_path)
        if score.get("status") != "C3_SAMPLE_SCORER_PASS" or score.get("sample_count") != 9:
            raise ValueError(f"Official scorer did not pass for {method}")
        if score.get("submission_sha256") != digest or run.get("submission_csv_sha256") != digest:
            raise ValueError(f"Scored CSV identity mismatch for {method}")
        if run.get("status") != "OFFICIAL_GENERATOR_AND_FORMULA_NINE_IMAGE_RESULT":
            raise ValueError(f"Run not complete for {method}")
        if run.get("evaluation_seed") != 42:
            raise ValueError(f"Expected seed 42 for {method}")
        row_mean = sum(number(row[f"{method}_row_score"], "row score") for row in rows) / 9
        block = summary[method]
        close(block["score_from_rows"], row_mean, f"{method} summary score")
        close(block["official_score"], row_mean, f"{method} official score")
        close(score["official_score_0_to_100"], row_mean, f"{method} scorer score")
        close(run["official_formula_score"], row_mean, f"{method} run score")
        for metric in METRICS:
            mean = sum(number(row[f"{method}_{metric}"], "metric") for row in rows) / 9
            close(block["means"][metric], mean, f"{method} {metric} mean")
        for term in TERMS:
            mean = sum(number(row[f"{method}_{term}"], "contribution") for row in rows) / 9
            close(block["contributions"][term], mean, f"{method} {term} mean")
        scored[method] = {"score": score, "run": run, "csv_sha256": digest}
    for method, baseline, key in ((PREVIOUS, CONTROL, "vs_control"),
                                  (CANDIDATE, CONTROL, "vs_control"),
                                  (CANDIDATE, PREVIOUS, "vs_stage21")):
        delta = [number(row[f"{method}_row_score"], "candidate row")
                 - number(row[f"{baseline}_row_score"], "baseline row") for row in rows]
        block = summary[method][key]
        close(block["score_delta"], sum(delta) / 9, f"{method} {key} score delta")
        close(block["median_row_score_delta"], median(delta), f"{method} {key} median")
        if block["images_improved"] != sum(value > 0 for value in delta):
            raise ValueError(f"Incorrect improved-image count in {method}/{key}")
        close(block["id2_row_score_delta"], delta[1], f"{method} {key} Id 2 delta")
    return summary, rows, scored


def visual_evidence(output: Path, rows: list[dict]):
    observations_path = output / "heatmaps/observations.json"
    observations = read_json(observations_path)
    manifest = read_json(output / "heatmaps/manifest.json")
    methods = ("stage21_16x16", "stage22_8x8")
    if set(manifest.get("methods", [])) != set(methods):
        raise ValueError("Both-method heatmaps are not complete")
    if manifest.get("observations_complete") is not True:
        raise ValueError("Both-method human observations are incomplete")
    if manifest.get("observations_sha256") != sha256(observations_path):
        raise ValueError("Heatmap manifest/observations hash mismatch")
    if manifest.get("stage22_raw_to_final_postprocess_verified") is not True:
        raise ValueError("Raw-to-final Stage 22 heatmap check is missing")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 18:
        raise ValueError("Expected 18 verified heatmap entries")
    lookup = {(item.get("method"), int(item.get("Id"))): item for item in entries}
    if len(lookup) != 18:
        raise ValueError("Duplicate heatmap method/Id")
    by_method = {}
    for method in methods:
        records = observations.get(method)
        if not isinstance(records, list) or len(records) != 9:
            raise ValueError(f"Expected nine observations for {method}")
        if [int(record["Id"]) for record in records] != list(range(1, 10)):
            raise ValueError(f"Observation Ids out of order: {method}")
        by_method[method] = records
        for index, record in enumerate(records, 1):
            entry = lookup.get((method, index))
            if entry is None or record.get("filename") != rows[index - 1]["image"]:
                raise ValueError(f"Heatmap source mismatch: {method}/Id {index}")
            if entry.get("observation") != record:
                raise ValueError(f"Heatmap manifest observation mismatch: {method}/Id {index}")
            for key in ("visual_observation_zh", "off_ship_heat_zh"):
                if not isinstance(record.get(key), str) or not record[key].strip():
                    raise ValueError(f"Missing {key}: {method}/Id {index}")
    for path_key in ("comparison_contact_sheet",):
        file = output / manifest[path_key]
        if not file.is_file():
            raise FileNotFoundError(file)
    return observations, manifest, by_method


def probe_evidence(output: Path) -> dict:
    probe = read_json(output / "probe.json")
    if probe.get("status") != "PASS" or probe.get("all_checks_passed") is not True:
        raise ValueError("Stage 22 probe did not pass")
    for key in ("original", "perturbed"):
        item = probe[key]
        if item.get("all_tile_masks_audited") != 2352 or item.get("selected_count_per_tile") != 64:
            raise ValueError(f"Incomplete 8×8 probe: {key}")
        if item.get("all_tiles_partition_complete_chw_once") is not True:
            raise ValueError(f"Incomplete CHW partition: {key}")
        if item.get("raw_attribution_finite") is not True or item.get("attribution_finite") is not True:
            raise ValueError(f"Nonfinite probe attribution: {key}")
    return probe


def paired_evidence(output: Path, required: bool) -> tuple[dict | None, list[dict] | None]:
    summary_path = output / "paired_summary.json"
    if not required:
        if summary_path.exists():
            raise ValueError("Paired summary exists though branch gate did not request repeats")
        return None, None
    paired = read_json(summary_path)
    records = paired.get("seeds")
    if not isinstance(records, list) or [int(record["seed"]) for record in records] != list(SEEDS):
        raise ValueError("Both predeclared paired seeds are required")
    csv_data = csv_rows(output / "paired_comparison.csv",
                        {"seed", "Id", "control_row_score", "candidate_row_score", "delta_row_score"},
                        18)
    for seed, record in zip(SEEDS, records, strict=True):
        group = [row for row in csv_data if int(row["seed"]) == seed]
        if len(group) != 9 or [int(row["Id"]) for row in group] != list(range(1, 10)):
            raise ValueError(f"Paired per-image rows incomplete for seed {seed}")
        deltas = []
        for row in group:
            delta = number(row["candidate_row_score"], "paired candidate row") - number(
                row["control_row_score"], "paired control row")
            close(row["delta_row_score"], delta, f"seed {seed} Id {row['Id']} delta")
            deltas.append(delta)
        close(record["control_score"], sum(number(row["control_row_score"], "paired control")
                                           for row in group) / 9, f"seed {seed} control")
        close(record["candidate_score"], sum(number(row["candidate_row_score"], "paired candidate")
                                             for row in group) / 9, f"seed {seed} candidate")
        close(record["delta_score"], sum(deltas) / 9, f"seed {seed} delta")
        close(record["median_row_delta"], median(deltas), f"seed {seed} median")
        if record["images_improved"] != sum(value > 0 for value in deltas):
            raise ValueError(f"Seed {seed} improved-image count changed")
        for stem, expected in ((f"control_seed{seed}", record["control_score"]),
                               (f"{CANDIDATE}_seed{seed}", record["candidate_score"])):
            source = output / "paired_runs" / stem
            score = read_json(source.with_suffix(".score.json"))
            run = read_json(source.with_suffix(".run.json"))
            raw_csv = source.with_suffix(".csv")
            actual_csv_sha = sha256(raw_csv)
            if (score.get("status") != "C3_SAMPLE_SCORER_PASS" or score.get("sample_count") != 9
                    or run.get("status") != "OFFICIAL_GENERATOR_AND_FORMULA_NINE_IMAGE_RESULT"
                    or run.get("evaluation_seed") != seed):
                raise ValueError(f"Paired official run incomplete: {stem}")
            if score.get("submission_sha256") != actual_csv_sha or run.get("submission_csv_sha256") != actual_csv_sha:
                raise ValueError(f"Paired CSV hash mismatch: {stem}")
            official_rows = csv_rows(raw_csv, {"Id", *METRICS}, 9)
            selected = "control_row_score" if stem.startswith("control_") else "candidate_row_score"
            for index, raw in enumerate(official_rows, 1):
                if int(raw["Id"]) != index:
                    raise ValueError(f"Paired official CSV Id mismatch: {stem}")
                f, m, s = (number(raw[metric], f"{stem}/{metric}") for metric in METRICS)
                value = 20 * (1 + f) + 30 * max(1 - m / 2, 0) + 30 * s
                close(group[index - 1][selected], value, f"Paired per-image formula {stem}/Id {index}")
            close(score["official_score_0_to_100"], expected, f"Paired score {stem}")
    return paired, csv_data


def source_record(environment: dict, name: str) -> dict:
    record = environment["files"][name]
    digest = record["sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Invalid source SHA-256: {name}")
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != digest:
        raise ValueError(f"Source file changed since environment inventory: {name}")
    return record


def markdown(output: Path, control_root: Path, stage21_root: Path) -> str:
    summary, rows, scored = scores_and_rows(output, control_root, stage21_root)
    observations, manifest, visual = visual_evidence(output, rows)
    probe = probe_evidence(output)
    environment = read_json(output / "environment.json")
    literature = (output / "literature.md").read_text(encoding="utf-8")
    rise_url = "https://arxiv.org/abs/1806.07421"
    if rise_url not in literature:
        raise ValueError("RISE primary paper reference missing from literature.md")
    # The scorer JSON is authoritative for a strict inequality. The expanded
    # row formula and summary are cross-checks, and can differ at the last bit.
    official_scores = {
        method: number(scored[method]["score"]["official_score_0_to_100"],
                       f"{method} original scorer score")
        for method in METHODS
    }
    candidate_score = official_scores[CANDIDATE]
    previous_score = official_scores[PREVIOUS]
    decision = read_json(output / "branch_decision.json")
    above = candidate_score > previous_score
    if decision.get("score_strictly_above_stage21") is not None and decision[
            "score_strictly_above_stage21"] is not above:
        raise ValueError("Branch decision disagrees with official scores")
    hotspot = decision.get("hotspots_roughly_ship")
    if above and not isinstance(hotspot, bool):
        raise ValueError("A score improvement requires a boolean human hotspot decision")
    if not above and hotspot is True:
        raise ValueError("Cannot promote when score gate failed")
    paired_required = above and hotspot is True
    paired, paired_rows = paired_evidence(output, paired_required)
    if summary[CANDIDATE]["strictly_above_stage21"] is not above:
        raise ValueError("Summary branch flag disagrees with official scores")
    candidate_run = scored[CANDIDATE]["run"]
    occlusion = candidate_run["occlusion"]
    cache = candidate_run["attribution_cache"]
    spec = cache["spec"]
    if occlusion.get("tile_size") != 8 or occlusion.get("tile_count_per_input") != 2352:
        raise ValueError("Unexpected scored 8×8 occlusion specification")
    if spec.get("tile_size") != 8 or spec.get("tile_area") != 64:
        raise ValueError("Unexpected cache specification")
    if candidate_run.get("evaluation_seed") != 42 or candidate_run.get("device") != "cpu":
        raise ValueError("Unexpected evaluation settings")
    for method in METHODS:
        if scored[method]["run"].get("official_weight_sha256") != source_record(
                environment, "weights")["sha256"]:
            raise ValueError(f"Weight hash changed for {method}")
        for run_key, source_key in (("official_generator", "official_generator"),
                                    ("official_scorer", "official_scorer")):
            if scored[method]["run"]["code_sha256"].get(run_key) != source_record(
                    environment, source_key)["sha256"]:
                raise ValueError(f"Official {run_key} hash changed for {method}")
    if candidate_run["code_sha256"].get("experiment") != source_record(
            environment, "stage22_code/experiment.py")["sha256"]:
        raise ValueError("Stage 22 scored code hash changed")
    if scored[PREVIOUS]["run"]["code_sha256"].get("experiment") != source_record(
            environment, "stage21_experiment")["sha256"]:
        raise ValueError("Stage 21 scored code hash changed")
    if scored[CANDIDATE]["csv_sha256"] != candidate_run["submission_csv_sha256"]:
        raise ValueError("Candidate CSV SHA-256 changed")
    if manifest.get("weight_sha256_from_scored_runs") != source_record(environment, "weights")["sha256"]:
        raise ValueError("Heatmap source weight hash differs")
    if manifest.get("program_sha256", {}).get("stage21_16x16") != source_record(
            environment, "stage21_experiment")["sha256"]:
        raise ValueError("Stage 21 heatmap program hash differs")
    if manifest.get("program_sha256", {}).get("stage22_8x8") != source_record(
            environment, "stage22_code/experiment.py")["sha256"]:
        raise ValueError("Stage 22 heatmap program hash differs")

    lines: list[str] = []
    add = lines.append
    add("# 第 22 階段：8×8 黑基準遮蔽與熱區驗證")
    add("")
    if paired_required:
        headline = "seed 42 分數嚴格高於 16×16，且人工熱區判為大致對應船舶；已依預先固定 seeds 完成成對重複。"
    elif above:
        headline = "seed 42 分數嚴格高於 16×16，但人工熱區未判為大致對應船舶；本輪不升為待驗證候選。"
    else:
        headline = "seed 42 分數未嚴格超過 16×16；依預定規則停止 tile 粒度家族。"
    add(f"**結論：{headline}** 8×8 九圖公式分數 **{fmt(candidate_score, 14)}**，16×16 為 **{fmt(previous_score, 14)}**，Mac 控制組為 **{fmt(official_scores[CONTROL], 14)}**。")
    add("")
    add("## 範圍、方法與來源")
    add("")
    add("所有分數均為 Mac 上主辦方提供的**九張圖**，由原版官方產生器及 `metrix.py`／`score_c3.py` 取得；並以 `20×(1+F)+30×max(1−M/2,0)+30×S` 獨立逐列重算。未公開圖片及正式平台分數仍未知；90 是研究目標，並非已確認的官方淘汰線。零基準 IG50 控制組與 16×16 使用先前同機 seed 42 的唯讀結果。")
    add("")
    add("8×8 對每次當前圖片（包括 Quantus 微擾圖）與傳入目標，取完整 CHW 的 `b=min(x)`；用指定類別原始與單通道單 tile 遮蔽後的 **logit** 差除以 64，帶符號回填。三通道共有 2,352 個不重疊 tile。後處理固定為完整 CHW 依絕對值前 25%、逐通道 reflect 5×5 平均、完整 CHW 依絕對值前 10%，均保留符號。無 SmoothGrad，且不依 Id、檔名或人工輪廓切換算法。seed 42、CPU、官方 batch 1、`shuffle=False`；遮蔽推論 batch 與執行緒設定見下文。")
    add("")
    for label, key in (("官方權重", "weights"), ("官方產生器", "official_generator"),
                       ("官方 `metrix.py`", "official_scorer"), ("第 22 階段歸因程式", "stage22_code/experiment.py"),
                       ("第 21 階段歸因程式", "stage21_experiment")):
        record = source_record(environment, key)
        add(f"- {label} SHA-256：`{record['sha256']}`" +
            (f"；{record['bytes']:,} bytes" if key == "weights" else "") + "。")
    add(f"- 8×8 官方 CSV SHA-256：`{scored[CANDIDATE]['csv_sha256']}`；原版計分器狀態：`{scored[CANDIDATE]['score']['status']}`。")
    add("")
    add("## 遮蔽探針與正式運行成本")
    add("")
    add("首張官方轉換後圖及固定細微擾動圖各檢查完整 2,352 個遮蔽輸入：每塊選中 64 個 CHW 位置、選中值填入各自當前圖的 `b`、未選位元組不變，整個 CHW 恰好覆蓋一次。首尾 tile 的獨立 logit 差分及原始／最終歸因形狀、`float32` 和有限性見 `probe.json`；狀態 `PASS`。")
    add("")
    add("| 探針輸入 | `b=min(x)` | 2,352 個遮蔽耗時 | 抽樣峰值 RSS |")
    add("|---|---:|---:|---:|")
    for label, key in (("首張原圖", "original"), ("固定細微擾動圖", "perturbed")):
        item = probe[key]
        benchmark = item["benchmark"]
        add(f"| {label} | {fmt(item['baseline_from_entire_chw'], 13)} | {fmt(benchmark['wall_seconds'], 3)} 秒 | {int(benchmark['rss_peak_sampled_bytes']):,} bytes |")
    add("")
    add(f"正式九圖產生器歷時 **{fmt(candidate_run['elapsed_seconds'], 3)} 秒**，歸因計算 **{fmt(occlusion['attribution_elapsed_seconds'], 3)} 秒**；實際計算 **{int(occlusion['computed_attributions'])}** 個不同當前輸入、**{int(occlusion['masked_inputs_evaluated']):,}** 個遮蔽輸入與 **{int(occlusion['model_forward_calls']):,}** 次模型 forward。遮蔽推論 batch **{int(occlusion['inference_batch_size'])}**、PyTorch 執行緒 **{int(occlusion['torch_num_threads_actual'])}**，峰值 RSS **{int(occlusion['peak_rss_bytes']):,} bytes**。快取寫入 **{int(cache['writes'])}**、磁碟命中 **{int(cache['disk_hits'])}**、記憶體命中 **{int(cache['memory_hits'])}**、無效分片 **{int(cache['invalid_entries'])}**。每筆分片同時保存帶符號原始圖與最終圖及兩個內容雜湊，依完整輸入／目標、權重、程式核心與設定、版本做內容定址。")
    add("")
    add("## 官方九圖分數與逐張分布")
    add("")
    add("| 方法（seed 42） | F 均值 | M 均值 | S 均值 | F 貢獻 | M 貢獻 | S 貢獻 | 九圖分數 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method, label in ((CONTROL, "Mac 零基準 IG50 控制組"),
                          (PREVIOUS, "16×16 黑基準遮蔽"),
                          (CANDIDATE, "8×8 黑基準遮蔽")):
        block = summary[method]
        values = [block["means"][metric] for metric in METRICS]
        terms = [block["contributions"][term] for term in TERMS]
        add("| " + label + " | " + " | ".join(fmt(value) for value in (*values, *terms))
            + f" | **{fmt(official_scores[method], 6)}** |")
    add("")
    current_f = number(summary[CANDIDATE]["means"]["Faithfulness"], "candidate F mean")
    current_m_term = number(summary[CANDIDATE]["contributions"][TERMS[1]],
                            "candidate sensitivity contribution")
    current_s_term = number(summary[CANDIDATE]["contributions"][TERMS[2]],
                            "candidate sparsity contribution")
    fixed_f_upper_bound = 20 * (1 + current_f) + 60
    needed_f_at_current_ms = (90 - current_m_term - current_s_term) / 20 - 1
    add(f"以本輪 8×8 的 F 均值固定，即使 M=0、S=1，公式分數上限為 **{fmt(fixed_f_upper_bound, 6)}**；若 M、S 維持本輪均值，要達 90，F 至少需 **{fmt(needed_f_at_current_ms, 6)}**。這只是公式條件，不代表該 F 值可由任何方法達成。")
    add("")
    add("| 比較（前者減後者） | 九圖分數差 | 逐圖差中位數 | 改善張數 | Id 2 逐圖差 | 其他八張差值合計 |")
    add("|---|---:|---:|---:|---:|---:|")
    for method, baseline, key, label in ((PREVIOUS, CONTROL, "vs_control", "16×16 − 控制組"),
                                         (CANDIDATE, CONTROL, "vs_control", "8×8 − 控制組"),
                                         (CANDIDATE, PREVIOUS, "vs_stage21", "8×8 − 16×16")):
        part = summary[method][key]
        deltas = [number(row[f"{method}_row_score"], "row") - number(row[f"{baseline}_row_score"], "row")
                  for row in rows]
        add(f"| {label} | {fmt(part['score_delta'], 6, True)} | {fmt(part['median_row_score_delta'], 6, True)} | {part['images_improved']}/9 | {fmt(deltas[1], 6, True)} | {fmt(sum(deltas) - deltas[1], 6, True)} |")
    add("")
    add("Id 2 欄為**單張逐圖分數差**，其他八張欄為八張差值總和；九張差值平均才是九圖分數差。這直接呈現增益是否集中於 Id 2，不能把平均分數改善等同於每張圖片都改善。M 越低對公式越有利。")
    add("")
    add("下表所有差值均為 **8×8 減比較方法**；`ΔM<0` 表示敏感度較低。完整精度和各項加權貢獻見 `per_image_comparison.csv`。")
    add("")
    add("| Id／圖片 | 8×8 F | 8×8 M | 8×8 S | 控制組逐圖分數 | 16×16 逐圖分數 | 8×8 逐圖分數 | Δ分數 vs 控制組 | Δ分數 vs 16×16 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        add(f"| {row['Id']}／{row['image']} | " + " | ".join(
            fmt(row[f"{CANDIDATE}_{metric}"]) for metric in METRICS) + " | "
            + " | ".join(fmt(row[f"{method}_row_score"]) for method in METHODS) + " | "
            + fmt(row[f"{CANDIDATE}_delta_row_score_vs_{CONTROL}"], signed=True) + " | "
            + fmt(row[f"{CANDIDATE}_delta_row_score_vs_{PREVIOUS}"], signed=True) + " |")
    add("")
    add("| Id | ΔF vs 控制組 | ΔM vs 控制組 | ΔS vs 控制組 | ΔF vs 16×16 | ΔM vs 16×16 | ΔS vs 16×16 |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        values = [row[f"{CANDIDATE}_delta_{metric}_vs_{baseline}"]
                  for baseline in (CONTROL, PREVIOUS) for metric in METRICS]
        add(f"| {row['Id']} | " + " | ".join(fmt(value, signed=True) for value in values) + " |")
    add("")
    add("## 兩種遮蔽粒度的熱圖與人工觀察")
    add("")
    add("[16×16 九圖聯絡表](heatmaps/stage21_16x16/contact_sheet.png) · [8×8 九圖聯絡表](heatmaps/stage22_8x8/contact_sheet.png) · [逐圖並列比較](heatmaps/comparison/contact_sheet.png)。熱圖從原圖輸入的已驗證快取分片還原，不依分片檔名排序猜測 Id。8×8 原始圖另核對固定後處理可逐位元組重現最終圖。")
    add("")
    add(f"顯示規則：{manifest['scale_rule']} 紅色為正、藍色為負。以下逐張記錄只作**事後目視**，沒有像素精確船舶真值；Id 7 舊手繪輪廓未完整包含貨櫃，不用它控制歸因或作合格證明。")
    add("")
    for index, row in enumerate(rows, 1):
        old = visual["stage21_16x16"][index - 1]
        new = visual["stage22_8x8"][index - 1]
        add(f"### Id {index} · {row['image']}")
        add("")
        add(f"- **16×16：**{old['visual_observation_zh']} 非船／邊界：{old['off_ship_heat_zh']}")
        add(f"- **8×8：**{new['visual_observation_zh']} 非船／邊界：{new['off_ship_heat_zh']}")
        comparison = new.get("comparison_zh")
        if isinstance(comparison, str) and comparison.strip():
            add(f"- **並列判讀：**{comparison.strip()}")
        add("")
    add("Id 5 的天空、Id 7 的天空／貨櫃、Id 8 的船與岸邊混合區應與上述各圖觀察及並列覆圖一起判讀；不能只以船體重疊外觀斷言解釋品質。")
    add("")
    add("## 預先指定的分支與後續")
    add("")
    if paired_required:
        add(f"8×8 seed 42 九圖分數 **{fmt(candidate_score, 14)}**，嚴格高於 16×16 的 **{fmt(previous_score, 14)}**；人工熱區判定為大致對應船舶。依預定規則標為**新的待驗證候選**，並對候選與零基準 IG50 控制組完成下列兩個 seed 的成對重複。16×16 只有 seed 42，不能推論其跨 seed 穩定性。")
        reason = decision.get("rationale_zh") or decision.get("reason_zh")
        if isinstance(reason, str) and reason.strip():
            add(f"熱區判斷理由：{reason.strip()}")
        add("")
        add("| seed | 控制組九圖分數 | 8×8 九圖分數 | 候選減控制組 | 逐圖差中位數 | 改善張數 |")
        add("|---:|---:|---:|---:|---:|---:|")
        assert paired is not None and paired_rows is not None
        for item in paired["seeds"]:
            add(f"| {item['seed']} | {fmt(item['control_score'])} | {fmt(item['candidate_score'])} | {fmt(item['delta_score'], signed=True)} | {fmt(item['median_row_delta'], signed=True)} | {item['images_improved']}/9 |")
        add("")
        add("| Id | seed 314159 逐圖差 | seed 271828 逐圖差 |")
        add("|---:|---:|---:|")
        for index in range(1, 10):
            values = [next(row for row in paired_rows if int(row["seed"]) == seed
                           and int(row["Id"]) == index)["delta_row_score"] for seed in SEEDS]
            add(f"| {index} | {fmt(values[0], signed=True)} | {fmt(values[1], signed=True)} |")
        add("")
        add("這兩次成對重複只提供主辦方九張圖上的 seed 變動線索；若少數 Id 集中貢獻淨增益，仍須在未見圖片與正式平台驗證。")
    elif above:
        add("8×8 的 seed 42 總分雖較高，但人工熱區未判為大致能對應船舶；未啟動配對 seeds，**不稱為穩定方法或新的待驗證候選**。")
        reason = decision.get("rationale_zh") or decision.get("reason_zh")
        if isinstance(reason, str) and reason.strip():
            add(f"熱區判斷理由：{reason.strip()}")
    else:
        add("8×8 的 seed 42 九圖分數未嚴格高於 16×16。依預先指定規則，停止 tile 粒度家族，不追加 4×4 或其他 tile／後處理掃描；配對 seeds 314159、271828 未啟動。16×16 仍只有 seed 42；兩法結果均不能據此宣稱跨 seed 穩定。")
    add("")
    add("## 論文線索與單一後續假說")
    add("")
    add("主辦方 Faithfulness 在完整 CHW 隨機抽 50 個**分散元素**，與連續單通道 8×8 tile 的 64 個位置不同；縮小 tile 是否有益只能由上述官方九圖實測判斷。指標背景見 [Bhatt 等，IJCAI 2020](https://www.ijcai.org/proceedings/2020/0417.pdf)；指標設計與限制見 [Hedström 等，JMLR 2023](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf)。論文不提供本次船舶九圖的成績。")
    add("")
    if not above:
        add(f"**下一個不同機制的單一可檢驗假說：**採用 [RISE 啟發的隨機重疊遮罩]({rise_url})，事先固定遮罩分布、數量、隨機種子、通道規則、黑基準、帶符號指定類別 logit 估計式與後處理，再以獨立原圖／微擾探針、官方九圖 F/M/S、逐張分布及熱區檢驗。重疊遮罩讓每個位置在多種遮蔽上下文中被估計，可能減輕整塊平均回填的限制；**這是由論文啟發的推論，不是已證實的提分方法，也不是原論文的原樣實作。**詳細權衡見 `literature.md`。")
    else:
        add(f"[RISE]({rise_url}) 等替代機制及成本列於 `literature.md`；本輪不因已知九圖結果改動預先固定的 8×8 方法。")
    add("")
    add("## 完整性與限制")
    add("")
    add(f"`heatmaps/manifest.json` 對九張圖逐一記錄官方原圖、轉換後輸入、目標、快取鍵、分片與歸因 SHA-256，兩法合計 **{len(manifest['entries'])}** 個原圖映射；人工觀察檔 SHA-256 為 `{sha256(output / 'heatmaps/observations.json')}`。`environment.json` 記錄官方來源、先前檢查點、程式與版本；`verification.json` 供獨立計分及內容檢查；`MANIFEST_SHA256.json` 與封裝程序供交付 ZIP 的 CRC／逐檔雜湊驗證。原圖熱區與逐圖分數只能描述這九張公開樣本；對未公開圖片及正式平台仍需上傳實測。")
    add("")
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--stage21-root", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    control_root = (args.control_root or output.parent / "output_mac_next").resolve(strict=True)
    stage21_root = (args.stage21_root or output.parent / "output_mac_stage21").resolve(strict=True)
    content = markdown(output, control_root, stage21_root)
    atomic_write(output / "report.md", content)
    print(json.dumps({"path": str(output / "report.md"), "bytes": len(content.encode("utf-8")),
                      "sha256": sha256(output / "report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
