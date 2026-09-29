# 第 22 階段：8×8 黑基準遮蔽與熱區驗證

本資料夾保存 `black_occlusion8_signed` 在主辦方提供九張圖片上的官方產生器實驗、與 Mac 控制組及第 21 階段 16×16 方法的配對比較，以及兩種遮蔽粒度的熱圖。九圖公式分數只描述這九張圖片；未公開圖片與正式平台結果尚未知。90 是研究目標，並非已證實的官方淘汰線。

## 固定方法

每次解釋只使用當前傳入的 `float32` NCHW 圖片、目標及官方模型，包括 Quantus 的微擾輸入。從完整三通道 CHW 取單一最小值 `b=min(x)`。對 3×224×224 的圖片，每通道切成 28×28 個不重疊 8×8 tile，合計 2,352 塊；一次只把一個通道的一塊 64 個位置替換為自己的 `b`。以指定類別未遮蔽與遮蔽後的 **logit** 差除以 64，帶符號回填；依完整 CHW 的絕對值保留前 25%，逐通道 reflect padding 作 5×5 平均，再依絕對值保留前 10%，保留符號。沒有 SmoothGrad，也不依 Id／檔名／手繪輪廓切換方法。

官方評估固定 seed 42、CPU、batch size 1、`shuffle=False`；各獨立遮蔽輸入以推論 batch 32、PyTorch 執行緒 8 加速。`experiment.py` 逐筆保存**後處理前的帶符號原始圖**和**最後歸因圖**，各有內容 SHA-256；快取鍵包含完整當前輸入與目標位元組、shape/dtype、官方權重 SHA-256、完整程式與遮蔽核心 SHA-256、參數及 PyTorch／NumPy 版本。寫入採原子檔案替換並在重載時驗證。中斷後可重用已驗證完整輸入分片，CSV 仍由原版官方產生器重新生成完整九列。

## 交付檔案

- `experiment.py`、`probe.py`／`probe.json`／`probe.log`：固定方法、雙輸入 2,352 塊遮蔽探針及實測時間／記憶體。
- `cache/black_occlusion8_signed/`：快取規格與原始／最終歸因分片。
- `runs/black_occlusion8_signed.{csv,score.json,run.json,log}`：官方原始九圖 CSV、原版計分器結果、執行統計與原始 log。
- `summarize.py`、`per_image_comparison.csv`、`summary.json`：控制組、16×16、8×8 的逐張指標及公式比較。
- `heatmaps.py`、`heatmaps/`：從已驗證的原圖輸入快取還原兩法九張原圖、覆圖、帶符號圖、聯絡表及逐圖人工觀察。人工觀察沒有像素精確的船舶真值。
- `paired_summary.py`、`paired_runs/`、`paired_summary.json`：若總分與熱區條件成立，保存固定 seeds 314159／271828 的候選與控制組成對重複。
- `literature.md`：原始論文、與官方指標的關係及下一輪可檢驗假說；不改動本輪固定候選。
- `build_report.py`、`report.md`：由原始計分與熱圖證據產生結果、決策及限制。`environment.json`、`verification.json`、`MANIFEST_SHA256.json`：來源、獨立核對與封包完整性。
- `instruction.md`：本輪工作指令的唯讀副本；原始檔路徑與雜湊另見環境紀錄。

## 重現及來源

在同一台 Mac 與相同官方檔案上，可由下列命令重做 seed 42；它會實際重新完成官方九圖產生器及原版計分器。避免覆寫交付成果時，請把 `--output-dir` 與 `--cache-dir` 指向新的資料夾。

```bash
cd '/Users/xiefujun/Downloads/AI研討會競賽規則(船舶影像辨識可解釋 AI)'
python3 output_mac_stage22/experiment.py \
  --stage17-root '/Users/xiefujun/Downloads/stage17' \
  --weights '開發環境/checkpoints/best_resnet50.pth' \
  --output-dir '/private/tmp/output_mac_stage22_recheck/runs' \
  --cache-dir '/private/tmp/output_mac_stage22_recheck/cache'
```

官方權重大小 282,678,575 bytes，SHA-256 `eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473`；官方產生器 SHA-256 `3f0c8830e590f6ad46934c034a5bf75f2d2b1d89baf2dba6683cdc1d9dc61df1`；`metrix.py` SHA-256 `5c89d3321c9328b47c4604ace0feed40d7cb9841843573d79b85838034ee2cda`。完整實際路徑、版本及來源檔雜湊見 `environment.json`。
