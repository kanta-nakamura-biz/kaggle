import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append('../../shared/src')

from data.data_loader import MitsuiDataLoader
from utils.visualization import (
    plot_time_series, plot_correlation_matrix, plot_distribution, 
    plot_missing_values, plot_target_analysis, save_plot
)

plt.rcParams["figure.figsize"] = (15, 8)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

def main():
    print("=== MITSUI&CO. Commodity Prediction Challenge - EDA ===")
    
    loader = MitsuiDataLoader('data/raw/')
    data = loader.load_competition_data()
    
    train_df = data['train']
    test_df = data['test']
    train_labels = data['train_labels']
    target_pairs = data['target_pairs']
    
    print(f"\n=== データ概要 ===")
    print(f"訓練データ: {train_df.shape}")
    print(f"テストデータ: {test_df.shape}")
    print(f"訓練ラベル: {train_labels.shape}")
    print(f"ターゲットペア: {target_pairs.shape}")
    
    lme_cols = [col for col in train_df.columns if 'LME_' in col]
    jpx_cols = [col for col in train_df.columns if 'JPX_' in col]
    us_cols = [col for col in train_df.columns if 'US_' in col]
    forex_cols = [col for col in train_df.columns if any(fx in col for fx in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD'])]
    
    print(f"\n=== 市場データ構成 ===")
    print(f"LME金属: {len(lme_cols)} columns")
    print(f"JPX先物: {len(jpx_cols)} columns") 
    print(f"US株式: {len(us_cols)} columns")
    print(f"外国為替: {len(forex_cols)} columns")
    
    print(f"\n=== 欠損値分析 ===")
    train_missing = train_df.isnull().sum()
    missing_cols = sum(train_missing > 0)
    print(f"欠損値のある列: {missing_cols}/{len(train_df.columns)}")
    print(f"最大欠損率: {(train_missing.max() / len(train_df) * 100):.1f}%")
    
    target_cols = [col for col in train_labels.columns if col.startswith('target_')]
    print(f"\n=== ターゲット変数分析 ===")
    print(f"ターゲット変数数: {len(target_cols)}")
    
    lag_dist = target_pairs['lag'].value_counts().sort_index()
    print(f"ラグ分布: {dict(lag_dist)}")
    
    all_targets = train_labels[target_cols].values.flatten()
    all_targets = all_targets[~np.isnan(all_targets)]
    
    print(f"\n=== 全ターゲット統計 ===")
    print(f"総データ点数: {len(all_targets):,}")
    print(f"平均: {np.mean(all_targets):.6f}")
    print(f"標準偏差: {np.std(all_targets):.6f}")
    print(f"最小値: {np.min(all_targets):.6f}")
    print(f"最大値: {np.max(all_targets):.6f}")
    
    print(f"\n=== 可視化作成中 ===")
    
    try:
        plot_missing_values(train_df, "訓練データ欠損値分析")
        save_plot('plots/missing_values_analysis.png')
        print("✅ 欠損値分析プロット保存完了")
    except Exception as e:
        print(f"❌ 欠損値分析プロット作成エラー: {e}")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(6):
            if i < len(target_cols):
                target_col = target_cols[i]
                target_data = train_labels[target_col].dropna()
                
                axes[i].hist(target_data, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{target_col}\nMean: {target_data.mean():.6f}, Std: {target_data.std():.6f}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('ターゲット変数分布 (最初の6つ)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_plot('plots/target_distributions.png')
        plt.show()
        print("✅ ターゲット分布プロット保存完了")
    except Exception as e:
        print(f"❌ ターゲット分布プロット作成エラー: {e}")
    
    try:
        lme_sample = lme_cols[:4] if len(lme_cols) >= 4 else lme_cols
        jpx_sample = [col for col in jpx_cols if 'Close' in col][:5]
        us_sample = [col for col in us_cols if 'close' in col][:10]
        forex_sample = forex_cols[:5] if len(forex_cols) >= 5 else forex_cols
        
        sample_cols = lme_sample + jpx_sample + us_sample + forex_sample
        sample_data = train_df[sample_cols].dropna()
        
        if len(sample_data) > 0 and len(sample_cols) > 1:
            plt.figure(figsize=(15, 12))
            corr_matrix = sample_data.corr()
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('市場間相関マトリックス (サンプル)', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_plot('plots/market_correlation_matrix.png')
            plt.show()
            print("✅ 市場間相関マトリックス保存完了")
    except Exception as e:
        print(f"❌ 相関マトリックス作成エラー: {e}")
    
    print(f"\n=== EDA完了 ===")
    print("次のステップ:")
    print("1. 特徴量エンジニアリング")
    print("2. モデル開発")
    print("3. 推論と提出")

if __name__ == "__main__":
    os.makedirs('plots', exist_ok=True)
    main()
