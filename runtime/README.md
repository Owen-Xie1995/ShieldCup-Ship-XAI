# 可獨立重跑的 4×4 soft15 解釋器

此目錄保存 2026-10-06 CPU 交付版本的原始部署程式與套件配方；`source_provenance.json` 記錄逐檔 SHA-256。與歷史交付檔逐位元相同的程式，不依賴第 23～26 階段的快取或歷史答案。

## 不需要競賽資料的示範

由倉庫根目錄依 [主 README](../README.md#快速執行) 安裝示範相依後：

```bash
python runtime/quick_demo.py
```

固定線性模型提供解析答案，檢查 supplied target 與 argmax 不同時仍解釋指定類別、正負 raw 歸因精確一致、FP32／NCHW 契約、輸入與 RNG 不變。此示範只需 NumPy、SciPy、PyTorch，沒有讀檔、下載模型或進行官方九圖評分。

直接使用解釋器：

```python
from runtime.src.explainer import custom_explain_func

attributions = custom_explain_func(model, inputs, targets, "cpu")
# inputs: finite NCHW, height/width multiples of 4
# targets: one integer class index per sample
# output: float32 NumPy array, same shape as inputs
```

baseline 是當次輸入張量的最小值。每次僅遮蔽一個通道中的 4×4 區塊，使用指定 target 的 logit drop，除以 16 並保留正負號。soft15 用每個樣本、每個通道的空間平滑幅度，取 85% quantile 作門檻，將連續 gate 乘回有號 raw。對 3×224×224 輸入共 9,408 個 masks，固定每批 64。

## 有主辦素材時重建完整 CPU 環境

先提供合法取得的官方目錄；不以 PyPI 的同名 Kaggle API 取代主辦 wheel。

```bash
python tools/prepare_local_runtime.py --official-dir /absolute/path/to/official --dry-run
python tools/prepare_local_runtime.py --official-dir /absolute/path/to/official
bash runtime/scripts/build.sh --platform linux/amd64
# ARM 主機可選 linux/arm64
```

準備工具只複製並核對指定權重、官方 wheel 和原始 scorer；這些路徑皆被 Git 忽略。資料檔與影像於執行時唯讀掛載，不需要放進映像。

```bash
bash runtime/scripts/run.sh \
  --image ship-xai:20261006-lock-cpu-amd64 \
  --task submission --data-dir /absolute/path/to/test_data \
  --output-dir /absolute/path/to/new_results -- \
  --csv-file /data/test.csv --images-dir /data/images \
  --output /output/submission.csv --device cpu --seed 42
```

runner 選用主辦 `val_transforms`，拒絕覆寫既有 CSV／執行證據，核對來源與有限值，保留失敗 checkpoint。有界記憶體快取以當次完整輸入、target、model state、device 與方法設定識別，沒有磁碟歷史答案。

## 實測範圍

2026-10-06 已實際完成 Linux ARM64 CPU 原生 VM、AMD64 CPU 模擬環境的建置／介面測試，以及 `docker save → load → smoke`。歷史證據與限制見 [部署驗收](../docs/deployment/README.md)。本次作品集整理沒有重建這兩個映像。

Docker 配方鎖定 CPU 版 Torch 2.13.0／torchvision 0.28.0；MPS 九圖歷史分數來自 Torch 2.10.0 的 Mac 環境。新 CPU 容器沒有完整九圖複評，亦沒有 CUDA／Jetson GPU 實測。`run.sh` 的 NVIDIA 掛載旗標需要另行提供相容的 GPU 映像；本配方建立的是 CPU 映像。
