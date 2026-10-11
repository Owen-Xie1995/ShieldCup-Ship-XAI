# CPU Docker 封裝與驗收

從 Mac MPS 研究程式整理成 Linux 可交付環境：獨立解釋器、來源 hash 綁定、固定相依、唯讀輸入掛載、拒絕覆寫結果，以及含 checksum 的映像匯出。

| 已完成的歷史檢查（2026-10-06） | 實測範圍 |
|---|---|
| 雙架構 CPU 建置 | ARM64 原生 Linux VM；AMD64 在同 VM 以模擬執行 |
| 套件／介面檢查 | 38 個套件版本、pip check、解析線性 toy |
| 官方模型小型 probe | ARM64 CPU 原圖與固定微擾各 128 masks；258 個樣本 forward |
| 舊 trace 重建 | 兩份完整 9,408-mask trace 的 raw／final FP32 位元一致；沒有新 inference |
| 映像交付 | 真正 docker save、docker load、兩架構載入後 smoke |
| 打包讀回 | 真正 7z test、新目錄解壓，160 個檔案集合／大小／SHA 核對 |

官方模型 probe 最大 target-drop relative L2 為 0.00010434937，小於事前 0.01 門檻。此結果不是完整九圖 Quantus 複評。

原九圖 CSV 保持 MPS seed42 的 82.65370613379287，沒有宣稱新的 CPU 九圖分數。CPU 映像不含 CUDA 版 PyTorch，亦沒有 Jetson GPU 執行證據。這個實驗把「可以封裝／載入」與「目標裝置的速度與數值等價」分開記錄。

## 證據與來源

`evidence/` 保存小型 JSON。為便於展示，已將本機使用者／絕對路徑改成佔位符；`provenance.json` 同時記錄原件與公開副本 hash。`runtime_acceptance.json` 內的來源 hash 指向原件，不能用來對經整理的 JSON 副本作逐位元檢查。

Docker tar、7z、官方權重／wheel／影像留在外部備份。此倉庫保存 [可重建配方](../../runtime/README.md) 與驗收範圍，而不包含這些大型交付物。
