# MITSUI&CO. Commodity Prediction Challenge - Kaggle競技

## 競技情報

- **プラットフォーム**: Kaggle
- **主催者**: MITSUI & CO., LTD.
- **競技ページ**: https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge
- **開始日**: 2025年8月
- **終了日**: 2025年10月6日
- **評価指標**: Sharpe ratio variant using Spearman rank correlation
- **賞金**: $100,000 USD

## 概要

商品価格の正確で安定した予測のための堅牢なモデルを開発する競技。
ロンドン金属取引所（LME）、日本取引所グループ（JPX）、米国株式、外国為替市場の過去データを使用して、
将来の商品リターンを予測することが課題。金融予測の精度向上と自動化された取引戦略の最適化が目標。

## データセット概要

[データセットの説明を記述]

## 評価指標

[評価指標の詳細説明]

## ディレクトリ構成

```
competitions/[競技名]/
├── data/               # 競技固有のデータ
│   ├── raw/           # 生データ
│   ├── processed/     # 前処理済みデータ
│   └── external/      # 外部データ
├── notebooks/          # EDA、モデリングノートブック
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_inference.ipynb
├── src/               # 競技固有のコード
│   ├── data/          # データ処理
│   ├── features/      # 特徴量エンジニアリング
│   ├── models/        # モデル定義
│   └── utils/         # ユーティリティ
├── models/            # 競技のモデル
├── submissions/       # 提出ファイル
├── configs/           # 競技の設定
└── logs/             # 競技のログ
```

## 実行手順

1. データの配置
2. EDAの実行
3. 特徴量エンジニアリング
4. モデル訓練
5. 推論と提出

## 結果

| 実験 | CV Score | Public LB | Private LB | 備考 |
|------|----------|-----------|------------|------|
| exp001 | 0.xxx | 0.xxx | 0.xxx | ベースライン |
| exp002 | 0.xxx | 0.xxx | 0.xxx | 特徴量追加 |

## 学んだこと

[競技を通じて学んだことを記録]

## TODO

- [ ] EDAの完了
- [ ] ベースラインモデルの作成
- [ ] 特徴量エンジニアリング
- [ ] モデルの改善
- [ ] アンサンブル
- [ ] 最終提出
