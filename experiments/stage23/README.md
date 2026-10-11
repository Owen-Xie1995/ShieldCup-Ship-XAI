# Stage23 — MPS correctness and RISE covariance negative result

這是歷史實驗的精選程式與小型證據，說明方法、事前門檻、實際成本及負面結果。實際結果與完成範圍見 [報告](../../docs/stage23/report.md)。本次整理未執行模型，也未重新認證歷史 cache。

- `src/`：作者撰寫的演算法、快取、預算或稽核程式快照；逐檔與原始來源位元組一致。
- `metadata/`：事前登記、決策、已保存的驗收／稽核與環境摘要；機器路徑以 `$WORKSPACE`、`$STAGE17` 或匿名 placeholder 表示。
- `results/`：實際 CSV、公式分數與逐圖配對差；sample/診斷分數的範圍見報告。
- `provenance.json`：來源相對路徑、原 SHA-256、公開副本 SHA-256 與轉換說明。檔案裡的歷史 hash 引用仍指原始證據，不因路徑匿名化假裝舊稽核被重跑。

`src/` 是供檢視的歷史快照，不是從此新位置可直接執行的套件。原始程式以 `output_mac_stage23` 為工作樹，部分依賴 machine-bound `common.py`、其他 stage、事前 source hash、未納入的 `.npz`／原始 JPEG／權重／官方 generator 與 scorer；這些來源需由使用者合法取得並核對。省略 machine-bound common／部分舊 runner、第三方 source、下載工具、原始資料、大型 cache、圖片、checkpoint、ZIP、完整原機器 manifest。`ROOT`／`parents` 相對目錄假設及凍結身份驗證未被改寫，重定位不能冒充同一歷史 run。可執行的專案入口以根目錄 `runtime/` 的說明為準。

保存的 PASS 表示當時完整本機證據的驗證狀態；本目錄未收錄全部被 hash 指向的檔案，因此不能在只有這份公開精選的環境重現全部歷史稽核。數字沒有外推為隱藏集資格分數或泛化證明。
