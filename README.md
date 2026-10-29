# Covertype 森林被覆タイプ分類

UCI Covertype データセット（581,012件・特徴量54列）を使い、標高・地形・土壌などの環境条件から、その土地を占める樹種（7クラス）を予測する多クラス分類。

## 構成

```
.
├── covertype/   # データ（git管理外）
├── notebooks/   # EDA
├── src/         # 学習・評価コード
└── outputs/     # 実行結果
```

## 実行方法

1. [UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype) から `covtype.data.gz` をダウンロードし、`covertype/` に置く
2. 実行する

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

動作確認環境：Python 3.11.15 / lightgbm 4.7.0 / scikit-learn 1.9.1 / pandas 3.1.0 / numpy 2.4.6 / matplotlib 3.11.2

## 検証方法

- 全データを train 80% / test 20% に層化分割（seed=42 で固定）
- train で StratifiedKFold による 5-fold CV を行う
- test は、5つの fold モデルの予測確率を平均して予測する
- 主指標は accuracy、副指標は macro F1

## 結果

| モデル | CV accuracy | test accuracy | test macro F1 |
|---|---|---|---|
| LightGBM ベースライン (問１) | 0.8549 ± 0.0033 | 0.8601 | 0.8503 |
| +正則化 | 0.8577 ± 0.0008 | 0.8597 | 0.8479 |