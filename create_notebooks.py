#!/usr/bin/env python3
"""
Jupyter notebook templates creation script
"""

import nbformat as nbf
import os

def create_eda_notebook():
    """EDAテンプレートノートブックを作成"""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell('# 探索的データ分析 (EDA) テンプレート\n\nこのノートブックは、Kaggle競技における探索的データ分析のテンプレートです。'),
        nbf.v4.new_markdown_cell('## 1. ライブラリのインポート'),
        nbf.v4.new_code_cell('import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings("ignore")\n\n# 設定\nplt.rcParams["figure.figsize"] = (12, 8)\npd.set_option("display.max_columns", None)'),
        nbf.v4.new_markdown_cell('## 2. データの読み込み'),
        nbf.v4.new_code_cell('# データの読み込み\ntrain = pd.read_csv("../data/raw/train.csv")\ntest = pd.read_csv("../data/raw/test.csv")\n\nprint(f"訓練データ形状: {train.shape}")\nprint(f"テストデータ形状: {test.shape}")'),
        nbf.v4.new_markdown_cell('## 3. 基本情報の確認'),
        nbf.v4.new_code_cell('# 基本情報の表示\nprint("=== 訓練データの基本情報 ===")\nprint(train.info())\nprint("\\n=== 訓練データの統計情報 ===")\nprint(train.describe())'),
        nbf.v4.new_markdown_cell('## 4. 欠損値の確認'),
        nbf.v4.new_code_cell('# 欠損値の確認\nprint("=== 訓練データの欠損値 ===")\nprint(train.isnull().sum())\nprint("\\n=== テストデータの欠損値 ===")\nprint(test.isnull().sum())'),
        nbf.v4.new_markdown_cell('## 5. ターゲット変数の分析'),
        nbf.v4.new_code_cell('# ターゲット変数の分布\nif "target" in train.columns:\n    plt.figure(figsize=(12, 4))\n    plt.subplot(1, 2, 1)\n    train["target"].hist(bins=50)\n    plt.title("Target Distribution")\n    plt.subplot(1, 2, 2)\n    train["target"].plot(kind="box")\n    plt.title("Target Box Plot")\n    plt.tight_layout()\n    plt.show()'),
        nbf.v4.new_markdown_cell('## 6. 特徴量の分析'),
        nbf.v4.new_code_cell('# 数値特徴量の相関行列\nnumeric_cols = train.select_dtypes(include=[np.number]).columns\nif len(numeric_cols) > 1:\n    plt.figure(figsize=(12, 10))\n    correlation_matrix = train[numeric_cols].corr()\n    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)\n    plt.title("Feature Correlation Matrix")\n    plt.show()'),
        nbf.v4.new_markdown_cell('## 7. カテゴリ変数の分析'),
        nbf.v4.new_code_cell('# カテゴリ変数の確認\ncategorical_cols = train.select_dtypes(include=["object"]).columns\nprint(f"カテゴリ変数: {list(categorical_cols)}")\n\nfor col in categorical_cols[:5]:  # 最初の5つのカテゴリ変数\n    print(f"\\n=== {col} ===")\n    print(train[col].value_counts().head(10))')
    ]
    
    os.makedirs('shared/notebooks', exist_ok=True)
    with open('shared/notebooks/01_EDA_template.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_feature_engineering_notebook():
    """特徴量エンジニアリングテンプレートノートブックを作成"""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell('# 特徴量エンジニアリング テンプレート\n\nこのノートブックは、Kaggle競技における特徴量エンジニアリングのテンプレートです。'),
        nbf.v4.new_markdown_cell('## 1. ライブラリのインポート'),
        nbf.v4.new_code_cell('import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder\nfrom sklearn.model_selection import train_test_split\nimport warnings\nwarnings.filterwarnings("ignore")'),
        nbf.v4.new_markdown_cell('## 2. データの読み込み'),
        nbf.v4.new_code_cell('# データの読み込み\ntrain = pd.read_csv("../data/raw/train.csv")\ntest = pd.read_csv("../data/raw/test.csv")\n\nprint(f"訓練データ形状: {train.shape}")\nprint(f"テストデータ形状: {test.shape}")'),
        nbf.v4.new_markdown_cell('## 3. 欠損値の処理'),
        nbf.v4.new_code_cell('# 欠損値の処理\n# 数値変数: 平均値で補完\nnumeric_cols = train.select_dtypes(include=[np.number]).columns\nfor col in numeric_cols:\n    if train[col].isnull().sum() > 0:\n        mean_val = train[col].mean()\n        train[col].fillna(mean_val, inplace=True)\n        test[col].fillna(mean_val, inplace=True)\n\n# カテゴリ変数: 最頻値で補完\ncategorical_cols = train.select_dtypes(include=["object"]).columns\nfor col in categorical_cols:\n    if train[col].isnull().sum() > 0:\n        mode_val = train[col].mode()[0]\n        train[col].fillna(mode_val, inplace=True)\n        test[col].fillna(mode_val, inplace=True)'),
        nbf.v4.new_markdown_cell('## 4. カテゴリ変数のエンコーディング'),
        nbf.v4.new_code_cell('# ラベルエンコーディング\nfrom sklearn.preprocessing import LabelEncoder\n\nle_dict = {}\nfor col in categorical_cols:\n    le = LabelEncoder()\n    # 訓練データとテストデータを結合してエンコーディング\n    combined = pd.concat([train[col], test[col]], axis=0)\n    le.fit(combined)\n    train[col + "_encoded"] = le.transform(train[col])\n    test[col + "_encoded"] = le.transform(test[col])\n    le_dict[col] = le\n    \nprint("ラベルエンコーディング完了")'),
        nbf.v4.new_markdown_cell('## 5. 数値変数の変換'),
        nbf.v4.new_code_cell('# 対数変換（正の値のみ）\nfor col in numeric_cols:\n    if (train[col] > 0).all():\n        train[col + "_log"] = np.log1p(train[col])\n        test[col + "_log"] = np.log1p(test[col])\n\n# 標準化\nscaler = StandardScaler()\nscaled_cols = [col for col in numeric_cols if col != "target"]\nif scaled_cols:\n    train_scaled = scaler.fit_transform(train[scaled_cols])\n    test_scaled = scaler.transform(test[scaled_cols])\n    \n    for i, col in enumerate(scaled_cols):\n        train[col + "_scaled"] = train_scaled[:, i]\n        test[col + "_scaled"] = test_scaled[:, i]'),
        nbf.v4.new_markdown_cell('## 6. 新しい特徴量の作成'),
        nbf.v4.new_code_cell('# 特徴量の組み合わせ\n# 例: 数値特徴量同士の四則演算\nif len(numeric_cols) >= 2:\n    col1, col2 = numeric_cols[0], numeric_cols[1]\n    train[f"{col1}_plus_{col2}"] = train[col1] + train[col2]\n    train[f"{col1}_minus_{col2}"] = train[col1] - train[col2]\n    train[f"{col1}_mult_{col2}"] = train[col1] * train[col2]\n    train[f"{col1}_div_{col2}"] = train[col1] / (train[col2] + 1e-8)\n    \n    test[f"{col1}_plus_{col2}"] = test[col1] + test[col2]\n    test[f"{col1}_minus_{col2}"] = test[col1] - test[col2]\n    test[f"{col1}_mult_{col2}"] = test[col1] * test[col2]\n    test[f"{col1}_div_{col2}"] = test[col1] / (test[col2] + 1e-8)'),
        nbf.v4.new_markdown_cell('## 7. 処理済みデータの保存'),
        nbf.v4.new_code_cell('# 処理済みデータの保存\nos.makedirs("../data/processed", exist_ok=True)\n\n# ターゲット変数の分離\nif "target" in train.columns:\n    X_train = train.drop("target", axis=1)\n    y_train = train["target"]\nelse:\n    X_train = train\n    y_train = None\n\nX_test = test\n\n# 保存\nX_train.to_csv("../data/processed/X_train_processed.csv", index=False)\nX_test.to_csv("../data/processed/X_test_processed.csv", index=False)\nif y_train is not None:\n    y_train.to_csv("../data/processed/y_train.csv", index=False)\n\nprint("処理済みデータを保存しました")\nprint(f"X_train形状: {X_train.shape}")\nprint(f"X_test形状: {X_test.shape}")\nif y_train is not None:\n    print(f"y_train形状: {y_train.shape}")')
    ]
    
    with open('shared/notebooks/02_feature_engineering_template.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_modeling_notebook():
    """モデリングテンプレートノートブックを作成"""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell('# モデリング テンプレート\n\nこのノートブックは、Kaggle競技におけるモデリングのテンプレートです。'),
        nbf.v4.new_markdown_cell('## 1. ライブラリのインポート'),
        nbf.v4.new_code_cell('import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import KFold, StratifiedKFold, cross_val_score\nfrom sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\nimport lightgbm as lgb\nimport xgboost as xgb\nimport catboost as cb\nfrom sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\nfrom sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso\nimport optuna\nimport joblib\nimport warnings\nwarnings.filterwarnings("ignore")\n\n# 設定\nplt.rcParams["figure.figsize"] = (12, 8)\npd.set_option("display.max_columns", None)\n\n# シード設定\nSEED = 42\nnp.random.seed(SEED)'),
        nbf.v4.new_markdown_cell('## 2. データの読み込み'),
        nbf.v4.new_code_cell('# 処理済みデータの読み込み\nX_train = pd.read_csv("../data/processed/X_train_processed.csv")\nX_test = pd.read_csv("../data/processed/X_test_processed.csv")\ny_train = pd.read_csv("../data/processed/y_train.csv").iloc[:, 0]\n\nprint(f"訓練データ形状: {X_train.shape}")\nprint(f"テストデータ形状: {X_test.shape}")\nprint(f"ターゲット形状: {y_train.shape}")'),
        nbf.v4.new_markdown_cell('## 3. ベースラインモデルの構築'),
        nbf.v4.new_code_cell('# LightGBMベースラインモデル\nfrom sklearn.model_selection import KFold\n\n# クロスバリデーション設定\nkf = KFold(n_splits=5, shuffle=True, random_state=SEED)\n\n# LightGBMパラメータ\nlgb_params = {\n    "objective": "regression",\n    "metric": "rmse",\n    "boosting_type": "gbdt",\n    "num_leaves": 31,\n    "learning_rate": 0.05,\n    "feature_fraction": 0.9,\n    "bagging_fraction": 0.8,\n    "bagging_freq": 5,\n    "verbose": -1,\n    "random_state": SEED\n}\n\n# クロスバリデーション実行\noof_predictions = np.zeros(len(X_train))\ntest_predictions = np.zeros(len(X_test))\ncv_scores = []\n\nfor fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):\n    print(f"Fold {fold + 1}")\n    \n    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]\n    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]\n    \n    # LightGBMデータセット作成\n    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)\n    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)\n    \n    # モデル訓練\n    model = lgb.train(\n        lgb_params,\n        train_data,\n        valid_sets=[val_data],\n        num_boost_round=1000,\n        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]\n    )\n    \n    # 予測\n    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)\n    test_pred = model.predict(X_test, num_iteration=model.best_iteration)\n    \n    oof_predictions[val_idx] = val_pred\n    test_predictions += test_pred / 5\n    \n    # スコア計算\n    fold_score = np.sqrt(mean_squared_error(y_val_fold, val_pred))\n    cv_scores.append(fold_score)\n    print(f"Fold {fold + 1} RMSE: {fold_score:.6f}")\n\n# 全体のスコア\noverall_score = np.sqrt(mean_squared_error(y_train, oof_predictions))\nprint(f"\\nOverall CV RMSE: {overall_score:.6f}")\nprint(f"CV RMSE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")'),
        nbf.v4.new_markdown_cell('## 4. 提出ファイルの作成'),
        nbf.v4.new_code_cell('# 提出ファイルの作成\nsubmission = pd.DataFrame({\n    "id": range(len(test_predictions)),  # 適宜IDカラムを調整\n    "target": test_predictions  # 適宜ターゲットカラム名を調整\n})\n\nos.makedirs("../submissions", exist_ok=True)\nsubmission.to_csv("../submissions/baseline_submission.csv", index=False)\nprint("提出ファイルを保存しました: ../submissions/baseline_submission.csv")\nprint(f"提出ファイル形状: {submission.shape}")\ndisplay(submission.head())')
    ]
    
    with open('shared/notebooks/03_modeling_template.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_inference_notebook():
    """推論テンプレートノートブックを作成"""
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell('# 推論・提出 テンプレート\n\nこのノートブックは、Kaggle競技における推論と提出ファイル作成のテンプレートです。'),
        nbf.v4.new_markdown_cell('## 1. ライブラリのインポート'),
        nbf.v4.new_code_cell('import pandas as pd\nimport numpy as np\nimport joblib\nimport os\nfrom datetime import datetime\nimport warnings\nwarnings.filterwarnings("ignore")\n\n# 設定\npd.set_option("display.max_columns", None)'),
        nbf.v4.new_markdown_cell('## 2. データとモデルの読み込み'),
        nbf.v4.new_code_cell('# テストデータの読み込み\nX_test = pd.read_csv("../data/processed/X_test_processed.csv")\nprint(f"テストデータ形状: {X_test.shape}")\n\n# 元のテストデータ（IDカラム用）\ntest_original = pd.read_csv("../data/raw/test.csv")\nprint(f"元のテストデータ形状: {test_original.shape}")\n\n# モデルの読み込み（例：joblib形式）\nmodel_dir = "../models/"\nif os.path.exists(model_dir):\n    model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]\n    print(f"利用可能なモデル: {model_files}")\n    \n    # 複数のモデルを読み込み（アンサンブル用）\n    models = {}\n    for model_file in model_files:\n        model_name = model_file.replace(".joblib", "")\n        models[model_name] = joblib.load(os.path.join(model_dir, model_file))\n        print(f"モデル読み込み完了: {model_name}")\nelse:\n    print("モデルディレクトリが見つかりません")'),
        nbf.v4.new_markdown_cell('## 3. 推論実行'),
        nbf.v4.new_code_cell('# 推論実行（モデルが読み込まれている場合）\nif "models" in locals() and models:\n    if len(models) == 1:\n        # 単一モデルでの推論\n        model_name = list(models.keys())[0]\n        model = models[model_name]\n        \n        print(f"単一モデル推論: {model_name}")\n        predictions = model.predict(X_test)\n        \n    else:\n        # 複数モデルのアンサンブル\n        print("アンサンブル推論")\n        all_predictions = []\n        \n        for model_name, model in models.items():\n            pred = model.predict(X_test)\n            all_predictions.append(pred)\n            print(f"{model_name}の推論完了")\n        \n        # 平均アンサンブル\n        predictions = np.mean(all_predictions, axis=0)\n        print("アンサンブル完了（平均）")\n    \n    print(f"予測値の形状: {predictions.shape}")\n    print(f"予測値の統計: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}")\nelse:\n    print("モデルが読み込まれていません。ダミーの予測値を作成します。")\n    predictions = np.random.random(len(X_test))'),
        nbf.v4.new_markdown_cell('## 4. 提出ファイルの作成'),
        nbf.v4.new_code_cell('# IDカラムの特定（適宜調整）\nid_column = "id"  # 競技に応じて変更\ntarget_column = "target"  # 競技に応じて変更\n\n# 提出ファイルの作成\nif id_column in test_original.columns:\n    submission = pd.DataFrame({\n        id_column: test_original[id_column],\n        target_column: predictions\n    })\nelse:\n    # IDカラムがない場合は連番で作成\n    submission = pd.DataFrame({\n        id_column: range(len(predictions)),\n        target_column: predictions\n    })\n\nprint(f"提出ファイル形状: {submission.shape}")\nprint("提出ファイルの最初の5行:")\ndisplay(submission.head())\n\nprint("\\n提出ファイルの最後の5行:")\ndisplay(submission.tail())'),
        nbf.v4.new_markdown_cell('## 5. 提出ファイルの保存'),
        nbf.v4.new_code_cell('# タイムスタンプ付きファイル名\ntimestamp = datetime.now().strftime("%Y%m%d_%H%M%S")\nfilename = f"submission_{timestamp}.csv"\nfilepath = f"../submissions/{filename}"\n\n# ディレクトリが存在しない場合は作成\nos.makedirs("../submissions", exist_ok=True)\n\n# ファイル保存\nsubmission.to_csv(filepath, index=False)\nprint(f"提出ファイルを保存しました: {filepath}")\n\n# 最新の提出ファイルとしてもコピー\nlatest_filepath = "../submissions/latest_submission.csv"\nsubmission.to_csv(latest_filepath, index=False)\nprint(f"最新提出ファイルを保存しました: {latest_filepath}")')
    ]
    
    with open('shared/notebooks/04_inference_template.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    print("Jupyter notebook templates作成中...")
    create_eda_notebook()
    print("✅ 01_EDA_template.ipynb 作成完了")
    
    create_feature_engineering_notebook()
    print("✅ 02_feature_engineering_template.ipynb 作成完了")
    
    create_modeling_notebook()
    print("✅ 03_modeling_template.ipynb 作成完了")
    
    create_inference_notebook()
    print("✅ 04_inference_template.ipynb 作成完了")
    
    print("🎉 全てのJupyter notebook templates作成完了!")
