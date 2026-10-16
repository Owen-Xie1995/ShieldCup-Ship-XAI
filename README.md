# ShieldCup Ship XAI

使用競賽提供的 ResNet50 船舶分類模型，比較影像歸因方法，並整理實驗結果、CPU Docker 環境與提交包檢查工具。分類模型與權重由主辦提供；這裡的工作集中在解釋方法、實驗和交付。

## 實驗結果

單通道 4×4 遮蔽搭配 soft15 後處理，在保存的兩組 Mac MPS 配對實驗中，分別比 8×8 對照高出 4.03 與 5.61 分。

![同九張影像、兩個評估種子的配對分數](docs/assets/paired-results.png)

| 評估 seed | 8×8 對照 | 4×4 soft15 | 分數差 | 改善張數 |
|---|---:|---:|---:|---:|
| 42 | 78.621952 | 82.653706 | +4.031754 | 9/9 |
| 314159 | 77.155063 | 82.761703 | +5.606641 | 9/9 |

兩組使用相同九張影像，換 seed 並沒有增加樣本，結果不能代表隱藏集表現。分數由保存的 CSV 重算；逐圖資料與方法見 [Stage25](docs/stage25/report.md)、[Stage26](docs/stage26/report.md)。2×2 方法只完成一張影像，未完成九圖評估。

## 執行示範

需要 Python 3.12。以下使用合成資料與固定線性模型，不需要競賽影像、權重或 GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate

# macOS
python -m pip install torch==2.10.0
# Linux 若只需要 CPU，改用：
# python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements-demo.txt
python runtime/quick_demo.py
python tools/check_saved_results.py
python tools/jetson_audit/test_tools.py -v
```

`quick_demo.py` 檢查指定類別、正負歸因、輸出形狀與 FP32 精度，也確認輸入及亂數狀態沒有被更動。指定類別刻意與模型最高分的類別不同，避免解釋錯誤目標。

![合成線性模型的輸入、原始歸因與後處理結果；顯示第三通道](docs/assets/synthetic-attribution.png)

這張圖顯示兩個合成輸入的第三通道（索引 2），不是船舶影像或競賽結果。圖表可重新產生：

```bash
python -m pip install -r requirements-figures.txt
python tools/build_demo_assets.py
```

## 程式與文件

| 要看什麼 | 入口 |
|---|---|
| 4×4 遮蔽與 soft15 實作 | [runtime/src/explainer.py](runtime/src/explainer.py) |
| 使用自己的模型與輸入 | [runtime/README.md](runtime/README.md) |
| Docker 建置、執行與歷史驗證紀錄 | [部署說明](docs/deployment/README.md) |
| CSV、manifest 與 Docker／OCI 映像檢查 | [驗收工具](tools/jetson_audit/README.md) |
| RISE 與梯度候選的失敗紀錄 | [Stage23](docs/stage23/report.md)、[Stage24](docs/stage24/report.md) |
| 方法取捨與展示順序 | [專案說明](docs/portfolio.md) |
| E017 團隊投稿與分工 | [研究案例](docs/publication/case-study.md)、[署名與來源](docs/publication/README.md) |

對 3×224×224 輸入，解釋器逐一遮蔽單一通道中的 4×4 區塊，共 9,408 個 masks，每批 64 個。它使用指定類別的 logit 降幅，保留正負號，並將區塊效果平均分給 16 個元素；這個分攤值不等於各元素的獨立因果效果。

`runtime/` 是目前可執行的獨立版本。`experiments/*/src` 與 `src/` 保存歷史研究程式，部分仍依賴原工作目錄和未公開素材。runner 會核對來源、保護既有輸出，並保留失敗紀錄。

## 驗證範圍與來源

已保存 Linux ARM64／AMD64 CPU 的建置、save／load／smoke 紀錄；CPU 容器尚未完成九圖複評，CUDA／Jetson GPU 尚未實測。CI 執行合成數值、檔案格式與映像檢查測試，不執行官方九圖評分。

本倉庫記錄 Mac 端研究、4×4 解釋器、CPU 部署、提交包驗收與團隊文件整理；開發過程使用 Codex 協作。E017 投稿算法與 runner 由隊友提供，論文是共同成果。正式競賽排名不在此處的驗證範圍內。

官方影像、權重、wheel、原始 scorer、含聯絡資料的文件與大型交付檔保留在本機或外部備份。完整環境的準備方式見 [runtime 操作說明](runtime/README.md)。
