# Stage25：4×4 遮蔽與 soft15 的 seed42 增益

固定單通道、非重疊 4×4 黑遮蔽，每張 9408 個 joint intervention，將 supplied-target logit 降幅除以 16 再均勻分配到 tile。這是集體干預的分配規則；沒有證明每個元素的個別精確效果。後處理只比較事前固定的 post22 與 soft15。

| seed42／主辦方提供九圖 | 公式均分 | 相對歷史同 MPS 8×8 78.621952211343 | 改善張數 |
| --- | ---: | ---: | ---: |
| 4×4 post22 | 81.473756017092 | +2.851803805749 | 8/9 |
| 4×4 soft15 | 82.653706133793 | +4.031753922450 | 9/9 |

兩法先完成事前前三圖篩選、各 3/3 改善，才進入完整九圖。選出 soft15，歷史決策為 GAIN_SEED42_ONLY_CONFIRMATION_PENDING；第二 seed、此候選額外公開五圖均未執行。對照來自核對來源的 Stage23 同 MPS seed42 九圖，本階段沒有重跑對照。

soft15 對 abs(raw) 進行 sigma=(0,1,1)、reflect Gaussian smoothing，以完整 CHW 0.85 分位數作 tau，乘回 signed raw 與 `max(1-tau/max(s,1e-12),0)`，沒有額外 top-k。post22 保留 abs top25%、reflect 5×5 平均、abs top10% 的 signed 值。

模型時計是單一持續帳：4826.173874/5400 秒，含來源／設備探針、正式指標、checkpoint 與 cache IO。post22 完整九圖流程 3175.345352 秒，其中新增 36 份冷 raw、另有前三圖的合法命中；soft15 17.422616 秒為全部 raw 暖快取但仍執行正式 Quantus model forwards。不得將兩者時間直接當成冷演算法速度比較。

MPS FP32 小量 CPU 子集驗收通過；完整 CPU 4×4 final 支持集未做。所有數字限於九張 sample 和 seed42；沒有泛化、90／95 分、官方資格或平台上傳證據。歷史 linear toy 證明 masking/reconstruction 的簽名和分配規則，不證明非線性 ResNet50 的元素 additivity。硬體是 MacBook Air M4；Windows CUDA 未驗證。這次整理只檢查文字與保存的 CSV，沒有重跑模型。

資料導航：[歷史程式與證據](../../experiments/stage25/README.md)，第二 seed 實際結果見 [Stage26](../stage26/report.md)。
