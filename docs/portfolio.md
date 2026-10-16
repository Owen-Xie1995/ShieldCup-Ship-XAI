# 船舶 XAI 專案說明

這個專案使用主辦提供的 ResNet50，處理兩個問題：如何解釋模型對指定船舶類別的分數，以及如何讓研究程式在另一個環境中重跑、檢查與交付。

## 主要工作

| 工作 | 可查閱內容 | 需要說清楚的限制 |
|---|---|---|
| 實作 4×4 有號遮蔽與 soft15 | [解釋器](../runtime/src/explainer.py)、[合成示範](../runtime/quick_demo.py) | 區塊內平均分攤不等於逐元素因果效果 |
| 比較候選方法與對照 | [Stage25](stage25/report.md)、[Stage26](stage26/report.md) | 兩個 seed 都使用同九張圖 |
| 整理 Linux CPU Docker 環境 | [部署紀錄](deployment/README.md) | CPU 驗證不能代替 Jetson GPU 實測 |
| 檢查 CSV、提交包與映像 | [驗收案例](jetson_audit/README.md) | 靜態檢查不能證明完整程式已在設備上執行 |
| 整理團隊研究與發表材料 | [研究案例](publication/case-study.md) | E017 算法與 runner 由隊友提供，論文共同署名 |

前期 RISE 與兩個梯度候選沒有通過比較門檻，結果保留在 [Stage23](stage23/report.md) 和 [Stage24](stage24/report.md)。後續 4×4 soft15 在兩組配對測試各提高 4.03／5.61 分，兩次都是 9/9 張改善；目前沒有獨立影像集可確認泛化效果。

## 展示順序

1. 用 [README 結果表](../README.md#實驗結果) 說明問題、對照與樣本範圍。
2. 執行 `python runtime/quick_demo.py`，展示指定類別與解析答案檢查。
3. 打開保存的逐圖 CSV、Docker 配方與驗收案例，說明怎麼追查結果和交付內容。

## 履歷描述

- 實作單通道 4×4 遮蔽與 soft15 歸因；在固定九圖、兩個評估 seed 的 Mac MPS 配對實驗中，較 8×8 對照分別提高 4.03 與 5.61 分。
- 整理 Linux ARM64／AMD64 CPU Docker 流程，保存依賴版本、來源 SHA-256 與 save／load／smoke 紀錄。
- 開發 CSV、提交包及 Docker／OCI 映像檢查工具，修正 Max_Sensitivity 範圍判斷，以邊界案例和映像測試驗證。

以上描述應與實際分工一同使用。倉庫目前為私有；提供履歷連結時，需要另外提供讀取權限。模型訓練、Jetson GPU 實測與正式排名未列入成果。
