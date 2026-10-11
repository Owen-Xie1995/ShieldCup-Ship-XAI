# Jetson 交付包驗收案例

這個案例呈現對船舶影像 XAI 交付包的獨立驗收：先核對來源與檔案雜湊，
再檢查容器內外檔案一致性，最後用合成資料重現 CSV admission 與 scorer
的合約差異。主要工程成果是找出一個會阻擋合法新資料的條件，並驗證
隊友修正版解除該條件；沒有宣稱 Jetson GPU 實測通過。

## 貢獻與來源

此處公開的是本專案工作區的驗收工具、測試案例及整理後的證據摘要，
由維護者透過 Codex 協作建立與整理。E017 解釋演算法與交付 runner
由隊友提供；模型權重、官方 wheel 與 scorer 由主辦方提供。此案例
展示驗收、缺陷定位與回歸驗證能力，沒有將隊友演算法列為維護者原創。

2026-10-08 至 2026-10-09 的完整 v1/v2 驗收紀錄提供了下表的歷史證據。
2026-10-11 整理公開工具時，再次對本機保留的兩版原始碼執行 AST
合成測試。原始交付包、競賽資料、權重、第三方程式、原始下載紀錄及
本機路徑未隨此案例公開；雜湊摘要可以核對當時審查的版本，公開
repository 本身不能重跑未提供的交付包。

## 發現與修正驗證

v1 的 admission 將 `Max_Sensitivity` 限制在 [0,2]，但隨附 scorer 對
大於 2 的有限值仍可評分，只將 sensitivity 項的貢獻截為 0。因此若
新資料产生合法有限值 3，v1 會在最終輸出完成前中止。這是條件式
複驗缺陷；沒有證據顯示原附件九列觸發該問題。

隊友 v2 將上界改為 `math.inf`，保留先檢查有限值與下界 0 的邏輯。
兩版 `strict_admission` 函式 AST 本體相同，差異來自 range 常數。
測試直接抽取經審閱的原函式及常數執行，沒有另寫替代 admission。

| 合成輸入 | v1 admission | v2 admission | scorer 行為 |
| --- | --- | --- | --- |
| sensitivity = 0、2 | 通過 | 通過 | 正常計分 |
| sensitivity = 2.000001、3、1e300 | 拒絕 | 通過 | sensitivity 項截為 0 |
| sensitivity 為負數、NaN、±Inf、非數字 | 拒絕 | 拒絕 | admission 阻擋 |
| Faithfulness 超出 [-1,1]、Sparsity 超出 [0,1] | 拒絕 | 拒絕 | admission 阻擋 |
| 欄位錯序、缺欄、多欄、列數或 Id 不符 | 拒絕 | 拒絕 | admission 阻擋 |

固定 Faithfulness=0.5、Sparsity=0.9 時，scorer 在 sensitivity 為
0、1、2、3、1e300 的五個案例得到預期的 87、72、57、57、57 分。
這是合成數值的函式測試；不是新模型分數或競賽正式成績。

## 實際證據與限制

| 檢查 | 2026-10-08/09 歷史結果 | 能支持的結論 |
| --- | --- | --- |
| 交付 archive SHA-256／7z 完整性 | 通過 | 當時下載檔與交付方摘要一致，archive 完整性檢查通過 |
| Manifest | 11/11 通過 | 記錄的檔案大小、SHA-256 與完整 inventory 一致 |
| Saved image layers | 69 層通過 | 檢查的 layer blob 與 rootfs digest 一致 |
| Image 與 package 選定檔案 | 7 檔一致 | 新 runner、README、weights 等選定檔案內外相符 |
| 核心檔案對 v1 | 6 檔未改變 | explainer、weights、CSV、wheel、scorer、dependencies 對當時 v1 manifest 一致 |
| Admission／scorer | 34/34、5/5 通過 | 合成數值、schema 與 Id 案例符合預期 |

公開工具整理後的驗證：八個小型合成 regression tests 通過，所有四個
Python 檔案語法通過；經審閱的 v1/v2 原始碼再次得到 admission
34/34、scorer 5/5 通過，測試程序没有 import runner 或 Torch，
沒有載入權重。環境為 macOS arm64、Python 3.12.9、NumPy 2.2.6、
pandas 3.0.3；與交付映像中的套件環境不同。

本輪工具檢查修正兩個合成案例的誤判：CSV 資料列多出欄位時會被
忽略，以及後續映像層把必要檔案替換成目錄時仍會保留舊檔案。
新增測試確認兩者均拒絕，也確認普通目錄項目仍保留下層子檔案。
這些是目前驗收工具的修正，不是對歷史交付包新增執行驗證。

驗收没有執行容器、完整 runner、正式九列重算或 Jetson GPU。無法由
以上結果認證 CUDA／ABI 相容性、記憶體、速度或跨裝置數值一致性。
ARM64 metadata 與正確檔案雜湊也無法替代裝置上的執行證據。

使用方法、依賴與 AST 執行邊界見 [tools README](../../tools/jetson_audit/README.md)。
精簡機器可讀證據見 [findings.json](findings.json)。

面試時可將此案例描述為：「對 XAI 交付包建立 manifest、容器層與
CSV 合約驗收，定位 sensitivity 上界與官方 scorer 不一致的缺陷，
並以 34 個 admission 案例與 5 個 scorer 案例驗證隊友修正版。」
