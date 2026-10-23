"""Run the CPU synthetic checks and recompute the saved nine-image comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# The demo reads existing sources and CSVs; it does not need bytecode files.
sys.dont_write_bytecode = True


def collect_report() -> dict:
    report = {
        "status": "FAIL",
        "scope": {
            "current_run": "Synthetic fixed linear model on CPU",
            "saved_results": "Historical Mac MPS CSVs; the same nine images for both seeds",
            "official_model_loaded": False,
            "competition_images_loaded": False,
            "new_nine_image_evaluation": False,
        },
        "current_cpu": {"status": "FAIL", "device": "cpu"},
        "saved_historical_mps": {"status": "FAIL"},
    }
    try:
        sys.path.insert(0, str(ROOT / "runtime" / "src"))
        import torch
        from smoke import run_toy

        toy = run_toy(torch.device("cpu"))
        if not all(toy["checks"].values()) or not all(toy["rng_unchanged"].values()):
            raise RuntimeError("CPU synthetic checks failed")
        report["current_cpu"].update(toy, status="PASS")
    except Exception as exc:
        report["current_cpu"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        from check_saved_results import check_readme, comparisons

        results = comparisons()
        check_readme(results)
        report["saved_historical_mps"].update(
            status="PASS", readme_matches=True, images_per_seed=9, comparisons=results
        )
    except Exception as exc:
        report["saved_historical_mps"]["error"] = f"{type(exc).__name__}: {exc}"

    if all(report[key]["status"] == "PASS" for key in ("current_cpu", "saved_historical_mps")):
        report["status"] = "PASS"
    return report


def print_summary(report: dict) -> None:
    print(f"{report['status']} | 面試示範")
    cpu = report["current_cpu"]
    print(f"\n本次 CPU 合成測試：{cpu['status']}")
    if cpu["status"] == "PASS":
        for sample in cpu["samples"]:
            print(f"  樣本 {sample['sample_index']}：指定類別 {sample['supplied_target']}，"
                  f"argmax {sample['argmax']}；原始歸因符合解析解")
        print("  PASS：指定類別、正負歸因、解析解、形狀 (2, 3, 8, 8)、FP32")
        print("  PASS：輸入與 target 未變動；Python／NumPy／PyTorch RNG 狀態未變動")
    else:
        print(f"  {cpu['error']}")

    saved = report["saved_historical_mps"]
    print(f"\n歷史 Mac MPS 結果（保存 CSV 重算、核對 README）：{saved['status']}")
    if saved["status"] == "PASS":
        print("    seed | 8×8 對照 | 4×4 soft15 | 分數差 | 改善張數")
        for row in saved["comparisons"]:
            print(f"  {row['seed']:6d} | {row['control']:.6f} | {row['candidate']:.6f} | "
                  f"{row['difference']:+.6f} | {row['images_improved']}/9")
    else:
        print(f"  {saved['error']}")
    print("\n範圍：本次只執行 CPU 合成測試與歷史 CSV 重算；"
          "未載入官方模型或競賽影像，也未重跑九圖評分。")
    print("兩個 seed 使用同九張影像，結果不能代表隱藏集表現。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the combined report as JSON")
    args = parser.parse_args()
    report = collect_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_summary(report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
