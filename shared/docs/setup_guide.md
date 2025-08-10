# セットアップガイド

## 環境構築

### 1. Python環境の準備

```bash
# Python 3.8以上を推奨
python --version

# 仮想環境の作成（推奨）
python -m venv kaggle_env
source kaggle_env/bin/activate  # Linux/Mac
# kaggle_env\Scripts\activate  # Windows
```

### 2. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

### 3. Kaggle APIの設定

1. Kaggleアカウントでログイン
2. Account → API → Create New API Token
3. `kaggle.json`をダウンロード
4. 適切な場所に配置：
   ```bash
   mkdir ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 4. 設定ファイルの準備

```bash
cp configs/config_template.yaml configs/config.yaml
# config.yamlを環境に合わせて編集
```

## 使用方法

### 新しい競技の開始

```bash
# テンプレートをコピー
cp -r competitions/template competitions/new-competition

# 競技データのダウンロード
cd competitions/new-competition
kaggle competitions download -c competition-name -p data/raw/
```

### ノートブックの実行

```bash
# Jupyter Labの起動
jupyter lab

# または特定のノートブックを実行
jupyter nbconvert --execute notebooks/01_eda.ipynb
```

## トラブルシューティング

### よくある問題

1. **Kaggle APIエラー**
   - `kaggle.json`の配置場所を確認
   - ファイルの権限を確認（600）

2. **ライブラリのインポートエラー**
   - 仮想環境がアクティブか確認
   - `pip install -r requirements.txt`を再実行

3. **メモリエラー**
   - データサイズを確認
   - バッチサイズを調整
   - 不要な変数を削除

## 推奨ワークフロー

1. EDAから開始
2. ベースラインモデルの作成
3. 特徴量エンジニアリング
4. モデルの改善
5. クロスバリデーション
6. アンサンブル
7. 最終提出
