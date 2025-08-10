#!/usr/bin/env python3
"""
Kaggle競技環境のセットアップスクリプト
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Python バージョンをチェック"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8以上が必要です")
        return False
    print(f"✅ Python {sys.version.split()[0]} を使用")
    return True


def install_requirements():
    """必要なライブラリをインストール"""
    print("📦 必要なライブラリをインストール中...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ ライブラリのインストール完了")
        return True
    except subprocess.CalledProcessError:
        print("❌ ライブラリのインストールに失敗しました")
        return False


def setup_kaggle_api():
    """Kaggle API の設定をチェック"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Kaggle API設定が見つかりました")
        return True
    else:
        print("⚠️  Kaggle API設定が見つかりません")
        print("以下の手順でKaggle APIを設定してください:")
        print("1. Kaggleアカウントでログイン")
        print("2. Account → API → Create New API Token")
        print("3. kaggle.jsonをダウンロード")
        print(f"4. {kaggle_dir}/ に配置")
        print("5. chmod 600 ~/.kaggle/kaggle.json を実行")
        return False


def create_config():
    """設定ファイルを作成"""
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("📝 設定ファイルをコピー中...")
        import shutil
        shutil.copy("configs/config_template.yaml", "configs/config.yaml")
        print("✅ config.yaml を作成しました")
    else:
        print("✅ config.yaml が既に存在します")


def setup_directories():
    """必要なディレクトリを作成"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models",
        "submissions",
        "logs",
        "experiments"
    ]
    
    print("📁 ディレクトリを作成中...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ ディレクトリの作成完了")


def create_gitkeep_files():
    """空のディレクトリに.gitkeepファイルを作成"""
    directories = [
        "data/raw",
        "data/processed",
        "data/external", 
        "models",
        "submissions",
        "logs",
        "experiments"
    ]
    
    for directory in directories:
        gitkeep_path = Path(directory) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
    
    print("✅ .gitkeepファイルを作成しました")


def main():
    """メイン関数"""
    print("🚀 Kaggle競技環境のセットアップを開始します\n")
    
    success = True
    
    if not check_python_version():
        success = False
    
    if not install_requirements():
        success = False
    
    if not setup_kaggle_api():
        print("⚠️  Kaggle APIの設定を後で行ってください")
    
    create_config()
    
    setup_directories()
    
    create_gitkeep_files()
    
    print("\n" + "="*50)
    if success:
        print("🎉 セットアップが完了しました!")
        print("\n次のステップ:")
        print("1. configs/config.yaml を編集して競技に合わせて設定")
        print("2. 新しい競技を始める場合:")
        print("   cp -r competitions/template competitions/your-competition-name")
        print("3. shared/notebooks/ のテンプレートを活用")
    else:
        print("❌ セットアップ中にエラーが発生しました")
        print("エラーを修正してから再度実行してください")
    
    print("="*50)


if __name__ == "__main__":
    main()
