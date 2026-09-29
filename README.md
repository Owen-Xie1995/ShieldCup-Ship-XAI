# ShieldCup Ship XAI

神盾盃船舶影像辨識可解釋 AI 競賽研究專案。

## Project Goal

針對競賽提供之 ResNet50 船舶影像分類模型，研究模型解釋方法，
並比較模型解釋的忠實度、敏感度、稀疏性及視覺呈現。

## Current Status

目前已完成至 Mac Stage 22 實驗階段。

現階段包含：

- ResNet50 模型推論與環境探測
- XAI 實驗流程
- heatmap 產生
- paired experiment summary
- environment / configuration recording
- report generation
- output verification

## Repository Policy

本 repository 不包含：

- 官方提供模型權重
- 官方競賽影像
- 官方測試資料
- 官方 wheel / binary package
- 私密 credential
- 大型實驗輸出

官方競賽素材僅保留於本機環境。

## Structure

```text
src/                    核心實驗與分析程式
experiments/            實驗設定與小型 metadata
docs/                   研究與開發紀錄
output_mac_stage*/      本機歷史 evidence，不納入 Git
開發環境/                官方競賽環境，不納入 Git