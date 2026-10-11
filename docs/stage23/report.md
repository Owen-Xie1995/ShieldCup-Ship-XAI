# Stage23：MPS 驗收與固定 RISE covariance 候選

本階段確認 MacBook Air M4 原生 MPS FP32 的歸因計算，並保留未通過門檻的候選。所有下列數值源自歷史記錄；這次整理沒有重跑模型。

| seed42／主辦方提供九圖 | 公式均分 | 相對同 MPS 8×8 | 改善張數 |
| --- | ---: | ---: | ---: |
| 8×8 signed 對照 | 78.621952211343 | — | — |
| RISE covariance 1024／14×14／post22 | 74.095659049856 | −4.526293161487 | 1/9 |

事前門檻為均分嚴格更高且至少 6/9 改善；候選 FAIL，保留 8×8。它是 RISE 啟發的獨立 signed covariance 變體，使用專用 PCG64(20260930) 固定遮罩 bank、當前完整 CHW 最低值作黑基準，並對每個新的 callback input 完整重算 1024 個遮罩。不是原版 RISE 的重現。

8×8 原圖／固定微擾的 CPU／MPS 探針分別比對 2353 個 target logits，零超限、raw 相對 L2 約 5.10e−5／6.23e−5，final 支持 IoU 均 1.0。完整歸因 CPU 187.107786／202.955990 秒、MPS batch32 18.030895／16.848203 秒；這些速度只適用這兩個探針。候選九圖 474.801346 秒、同設備 8×8 九圖 1039.298023 秒。

額外公開五圖已在評分前凍結，狀態為 FROZEN_NOT_SCORED_BUDGET、0/5 對，均分為 null。準備成本 3558/3600 秒是保守預算估計，精確主動準備時間未完整量測。Stage24 才完成該五圖評分。

公式是逐列 `20*(1+F)+30*max(1-M/2,0)+30*S` 再平均，使用未修改的主辦方 sample generator/scorer。九張 sample 不是隱藏集資格分數；沒有泛化、平台上傳或 Windows CUDA 驗證的證據。CPU 歷史單圖與 MPS 的 RNG reset 位置曾不同，因此九圖候選判定使用同 MPS 對照。

資料導航：[歷史程式與證據](../../experiments/stage23/README.md)。原報告來源與整理後檔案雜湊記錄於該目錄 provenance.json。
