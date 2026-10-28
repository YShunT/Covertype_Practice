```
src/
├── config.py    # パス・列名・クラス名・乱数シード・LightGBMパラメータ
├── data.py      # データ読み込み、ラベルを0〜6に変換、train/testに層化分割
├── models.py    # LightGBMの生成と学習
├── train.py     # 5-fold CV（OOF予測、test予測の5モデル平均）
├── evaluate.py  # 評価指標、混同行列・特徴量重要度・学習曲線の描画
└── main.py      # 実行の入口、ログとmetrics.json・図の保存
```
