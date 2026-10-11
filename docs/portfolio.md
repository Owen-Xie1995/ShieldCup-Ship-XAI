# 求職專案說明

## 專案定位

**船舶影像 XAI 實驗與部署工程**：使用固定 ResNet50，設計／比較輸入歸因方法，將 Mac 研究程式整理成 Linux CPU 容器，以可追溯的檔案、CSV 與映像驗收確認交付內容一致。

可展示 Python／PyTorch 數值程式、實驗設計、Docker、測試與技術文件能力。分類模型由主辦提供，沒有將使用預訓練權重表述成自行訓練分類器。

## 面試可展示內容

| 工程重點 | 實作證據 | 可以說明的取捨 |
|---|---|---|
| XAI 契約與數值正確性 | 解釋器、解析 toy | supplied target、FP32、正負號與 tile 分攤假設 |
| 配對實驗 | Stage25／26 CSV 及逐圖差 | 兩個 seed 各對各自對照；同九張圖的外推限制 |
| 可重建與交付 | Dockerfile、版本 pins、save／load receipt | CPU 交付與 Jetson GPU 執行是不同驗證範圍 |
| 輸出保護 | runner、有界快取、來源 hash | 防止誤用歷史答案、覆寫成果或把失敗 CSV 當完成 |
| 驗收工具 | package／image inspector 及合成測試 | 不執行完整 package 程式，檢查 layer／manifest／CSV |
| 問題追查 | Max_Sensitivity 邊界修正 | 評分會裁切貢獻的值，不應被 admission 錯拒 |
| 團隊溝通 | 研究摘要及精簡口述稿 | 說清方法來源、共同作者和實測限制 |

## 履歷可用敘述

以下對應倉庫可查證工作，使用時依自己的實際角色調整：

- 研究並實作單通道 4×4 有號遮蔽與 soft15 歸因；固定九圖、兩個評估 seed 的 Mac MPS 配對測試中，較 8×8 對照提高 4.03／5.61 分，兩次皆 9/9 張改善。
- 整理 Linux ARM64／AMD64 CPU Docker 交付流程，保存固定相依、來源 SHA-256、save／load／smoke 與小型官方模型 probe 證據。
- 開發提交包與 Docker／OCI 映像驗收工具，追查 CSV admission 對 Max_Sensitivity 的範圍契約缺陷，以邊界案例和 manifest／layer 測試驗證修正。

Jetson 實測、完整 2×2 九圖、隱藏集泛化、自行訓練分類模型與正式競賽排名均超過目前證據。E017 由隊友提供，投稿論文保持共同作者歸屬。

## 3 分鐘展示順序

1. 從 [主 README](../README.md) 的結果表說明問題、對照與限制。
2. 執行 `python runtime/quick_demo.py`，說明指定 target 為何不同於 argmax，以及解析答案如何驗證實作。
3. 展示配對 CSV、Docker 配方及 [驗收案例](jetson_audit/README.md)，說明研究、交付和設備實測範圍。

倉庫原本是私有專案。提供履歷連結前需給招募者存取權，或另行決定公開；本次整理保留既有可見性。
