# 第 22 階段相關論文與下一輪可檢驗假設

查閱日期：2026-09-29。本文只提供研究背景與**下一輪**實驗方向；不變更第 22 階段已預先固定的 `black_occlusion8_signed`，也不把其他資料集的論文結果當成這九張船舶圖片的成績。

## 先按實際評分程式讀問題

主辦方原版 `submmit.py` 建立 Quantus `FaithfulnessCorrelation(subset_size=50, perturb_baseline="black")`、`MaxSensitivity(nr_samples=5, lower_bound=-0.01, upper_bound=0.01)` 及 `Sparseness()`。Mac 已安裝的 Quantus `faithfulness_correlation.py` 預設 `nr_runs=100`、`abs=False`、`normalise=True`；每次從 3×224×224＝150,528 個 CHW 位置中**不重複抽 50 個分散元素**，換成黑基準，將這 50 個位置的歸因和，與原始及擾動後**指定類別 logit 的差**算 Pearson 相關。這個評分擾動與遮住單一連續的 8×8 或 16×16 tile 不同。其原始評估論文是 [Bhatt、Weller、Moura，IJCAI 2020](https://www.ijcai.org/proceedings/2020/0417.pdf)；Quantus 的指標分類與侷限可參考 [Hedström 等，JMLR 2023](https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf)。

第 21 階段 16×16 的九圖 F 均值僅 0.114524、官方公式分數 76.052595，對控制組總分 +0.465015，但逐圖只有 4/9 改善。第 22 階段縮到 8×8 是**實驗假設**，不能從 64 接近 50 就推出它會提高相關：64 是連續且同通道的 tile，評分的 50 個位置分布於完整 CHW。官方分數同時獎勵 F、低 MaxSensitivity 與稀疏度；任何候選應報三項及逐圖分布，而非只顯示熱圖。關於對微擾分布選擇敏感的解釋評估，可參考 [Yeh 等，NeurIPS 2019](https://arxiv.org/abs/1901.09392)。

## 可能機制與實驗成本

| 機制 | 原始證據 | 對這個評分的可檢驗推論與風險 | 約略成本／優先序 |
|---|---|---|---|
| **隨機重疊遮罩，估計像素的邊際影響** | [RISE，Petsiuk、Das、Saenko，BMVC 2018](https://arxiv.org/abs/1806.07421) 以隨機遮罩和模型輸出估計像素顯著性，原文在其他影像資料集比較刪除／插入及定位。 | 遮罩重疊、覆蓋全圖，可觀察特徵交互，避免把一塊的 logit 差平均回填所有 64 個位置。**推論：**用各通道獨立的隨機遮罩、當前圖自己的黑基準、指定類別 logit，並估計帶符號的「保留 vs 遮去」輸出差，可能改善 F 或熱區。這是 **RISE 啟發的改法，並非原論文方法**；原法用類別機率且常用通道共用的空間遮罩。大面積隨機遮蔽會產生離分布影像，且與評分時分散抽 50 個元素也不相同，故正負結果必須實測。 | 高；若下一輪預先固定 2,048 個遮罩，每次歸因的前向次數與本輪 2,352 個 8×8 tile 同量級，但實際耗時需先探針。**建議下一個單一機制**，僅在本輪未優於 16×16 時考慮。 |
| **直接匹配評分擾動分布的歸因擬合** | [Yeh 等，NeurIPS 2019](https://arxiv.org/abs/1901.09392) 指出歸因的 infidelity 依擾動分布而變，並研究平滑與敏感度；[Bhatt 等](https://www.ijcai.org/proceedings/2020/0417.pdf) 給出隨機特徵子集的 Faithfulness 框架。 | 可用預先固定的、與官方 50 個 CHW 位置同分布的黑基準擾動，將 logit 差對歸因和做受正則化擬合；再在**獨立圖片／seed**驗證。這是由論文引出的新方案，並非兩篇論文已證明能提升本競賽分數。150,528 個未知特徵極高維，50 個位置只佔 0.0332%，足夠覆蓋與辨識的採樣成本可能過高；直接用公開九張圖擬合又會嚴重過擬合。 | 很高；先做樣本數與辨識度可行性分析，暫不作下一輪首選。 |
| **梯度隨機集成的平方／變異量** | [SmoothGrad，Smilkov 等，2017](https://arxiv.org/abs/1706.03825) 改善梯度圖的視覺噪聲；[Hooker 等，NeurIPS 2019](https://arxiv.org/abs/1806.10758) 在其 ROAR 基準中發現 VarGrad 與 SmoothGrad-Squared 表現較好，並提醒集成方式及成本重要。 | 第 20 階段已測「五次噪聲的**平均**梯度或 IG」，不等於平方／變異量；後者是未測的不同統計量。它可能穩定細節，但非負值會抹掉抑制指定類別的負向貢獻，與目前 Faithfulness 的 `abs=False` 不一定相合。論文的 ROAR 與本競賽公式不同。 | 中；可用有限次反向傳播先做探針，但因符號與既往平均梯度結果，排在隨機遮罩之後。 |
| **最佳化連續遮罩／極端擾動** | [Fong、Vedaldi，ICCV 2017](https://arxiv.org/abs/1704.03296) 用有意義的影像擾動學習遮罩；[Fong、Patrick、Vedaldi，ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Fong_Understanding_Deep_Networks_via_Extremal_Perturbations_and_Smooth_Masks_ICCV_2019_paper.html) 進一步使用面積約束與平滑遮罩。 | 可能帶來更整齊的船體熱區，但整片區域遮蔽、平滑約束與官方 50 個分散 CHW 元素的 Faithfulness 之間仍有落差。背景 sky／岸邊若模型真正使用，也不能為了視覺整齊而抹除。 | 高；需要逐次最佳化及額外檢查遮罩所造成的影像假訊號，暫不優先。 |
| **Shapley／KernelSHAP 特徵歸因** | [Lundberg、Lee，NeurIPS 2017](https://papers.nips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) 建立加性特徵歸因框架；[Integrated Gradients，Sundararajan、Taly、Yan，ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf) 則說明路徑積分與基準選擇。 | Shapley 可處理交互，但對 150,528 個 CHW 元素直接估計不可行；用超像素降低成本會再次面對「群組 attribution 對分散 50 元素」的對齊問題。零基準 IG50 已是 Mac 控制組，黑基準 IG50 變體在第 20 階段也未超過控制組；重做同一組合缺乏新假設。 | 極高或已測；目前不列為下一輪優先。 |

## 若 8×8 未超過 16×16：建議下一個**單一**預註冊方案

停止 tile 粒度家族後，選 **RISE 啟發的帶符號隨機重疊遮罩**。實作前一次固定遮罩分布、每張遮罩數、隨機種子、各通道是否獨立、黑基準、指定類別 logit 的帶符號估計式、後處理和快取版本；先做原圖及細微擾動圖的遮罩／耗時探針，再以官方九圖與 Mac 控制組、16×16、8×8 的同一公式比較 F／M／S、九張差值及 Id 2 增益集中度。若九圖總分上升，依事先固定 seed 做成對重複與熱區檢查。這個方案之所以值得**測**，是它把各像素在多種遮蔽上下文中的邊際效果分開估計；目前沒有論文或本地實驗能保證船舶資料的得分會提高。

熱圖僅作解釋品質線索。[Adebayo 等，NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) 顯示單靠視覺判斷會誤導；船體重疊、刪除／插入、輕微平移與逐圖官方分數都應一起閱讀，且不得把公開九張圖的人工輪廓用於歸因或選參。
