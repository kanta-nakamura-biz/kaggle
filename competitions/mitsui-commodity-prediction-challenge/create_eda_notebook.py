import nbformat as nbf
import json

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    setup_code = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.append('../src')
sys.path.append('../../shared/src')

from data.data_loader import MitsuiDataLoader
from utils.visualization import (
    plot_time_series, plot_correlation_matrix, plot_distribution, 
    plot_missing_values, plot_target_analysis, save_plot
)

plt.rcParams["figure.figsize"] = (15, 8)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

print("=== MITSUI&CO. Commodity Prediction Challenge - EDA ===")'''
    
    nb.cells.append(nbf.v4.new_code_cell(setup_code))
    
    data_loading_code = '''# Load competition data using custom loader
loader = MitsuiDataLoader('../data/raw/')
data = loader.load_competition_data()

summary = loader.get_data_summary(data)
print("データ概要:")
for name, info in summary.items():
    print(f"\\n{name.upper()}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Memory: {info['memory_usage']/1024/1024:.1f} MB")
    if info['missing_values']:
        missing_count = sum(1 for v in info['missing_values'].values() if v > 0)
        print(f"  Missing values: {missing_count} columns")

issues = loader.validate_data_structure(data)
if issues:
    print(f"\\nデータ構造の問題: {issues}")
else:
    print("\\nデータ構造検証: 正常")'''
    
    nb.cells.append(nbf.v4.new_code_cell(data_loading_code))
    
    market_analysis_code = '''# Extract individual datasets
train_df = data['train']
test_df = data['test']
train_labels = data['train_labels']
target_pairs = data['target_pairs']

print(f"訓練データ: {train_df.shape}")
print(f"テストデータ: {test_df.shape}")
print(f"訓練ラベル: {train_labels.shape}")
print(f"ターゲットペア: {target_pairs.shape}")

lme_cols = [col for col in train_df.columns if 'LME_' in col]
jpx_cols = [col for col in train_df.columns if 'JPX_' in col]
us_cols = [col for col in train_df.columns if 'US_' in col]
forex_cols = [col for col in train_df.columns if any(fx in col for fx in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD'])]

print(f"\\n市場データ構成:")
print(f"  LME金属: {len(lme_cols)} columns")
print(f"  JPX先物: {len(jpx_cols)} columns") 
print(f"  US株式: {len(us_cols)} columns")
print(f"  外国為替: {len(forex_cols)} columns")

print(f"\\nLME金属データ: {lme_cols}")
print(f"JPX先物データ例: {jpx_cols[:10]}")
print(f"US株式データ例: {us_cols[:10]}")'''
    
    nb.cells.append(nbf.v4.new_code_cell(market_analysis_code))
    
    missing_analysis_code = '''# Missing values analysis
print("=== 欠損値分析 ===")

train_missing = train_df.isnull().sum()
train_missing_pct = (train_missing / len(train_df)) * 100
train_missing_df = pd.DataFrame({
    'Column': train_missing.index,
    'Missing_Count': train_missing.values,
    'Missing_Percentage': train_missing_pct.values
}).sort_values('Missing_Count', ascending=False)

print(f"訓練データ欠損値統計:")
print(f"  完全な列: {sum(train_missing == 0)} columns")
print(f"  欠損値のある列: {sum(train_missing > 0)} columns")
print(f"  最大欠損率: {train_missing_pct.max():.1f}%")

plot_missing_values(train_df, "訓練データ欠損値分析")

print("\\n欠損値の多い列 (上位20):")
print(train_missing_df.head(20))'''
    
    nb.cells.append(nbf.v4.new_code_cell(missing_analysis_code))
    
    target_analysis_code = '''# Target analysis
print("=== ターゲット変数分析 ===")

print("ターゲットペア構造:")
print(target_pairs.head(10))

lag_dist = target_pairs['lag'].value_counts().sort_index()
print(f"\\nラグ分布:")
print(lag_dist)

target_cols = [col for col in train_labels.columns if col.startswith('target_')]
print(f"\\nターゲット変数数: {len(target_cols)}")

target_stats = train_labels[target_cols].describe()
print("\\nターゲット変数統計 (最初の5つ):")
print(target_stats.iloc[:, :5])

target_missing = train_labels[target_cols].isnull().sum()
print(f"\\nターゲット変数欠損値:")
print(f"  欠損値のないターゲット: {sum(target_missing == 0)}")
print(f"  欠損値のあるターゲット: {sum(target_missing > 0)}")'''
    
    nb.cells.append(nbf.v4.new_code_cell(target_analysis_code))
    
    target_viz_code = '''# Visualize target distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i in range(6):
    if i < len(target_cols):
        target_col = target_cols[i]
        target_data = train_labels[target_col].dropna()
        
        axes[i].hist(target_data, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{target_col}\\nMean: {target_data.mean():.6f}, Std: {target_data.std():.6f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

plt.suptitle('ターゲット変数分布 (最初の6つ)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

all_targets = train_labels[target_cols].values.flatten()
all_targets = all_targets[~np.isnan(all_targets)]

print(f"\\n全ターゲット統計:")
print(f"  総データ点数: {len(all_targets):,}")
print(f"  平均: {np.mean(all_targets):.6f}")
print(f"  標準偏差: {np.std(all_targets):.6f}")
print(f"  最小値: {np.min(all_targets):.6f}")
print(f"  最大値: {np.max(all_targets):.6f}")'''
    
    nb.cells.append(nbf.v4.new_code_cell(target_viz_code))
    
    correlation_code = '''# Market correlation analysis
print("=== 市場間相関分析 ===")

lme_sample = lme_cols[:4] if len(lme_cols) >= 4 else lme_cols
jpx_sample = [col for col in jpx_cols if 'Close' in col][:5]
us_sample = [col for col in us_cols if 'close' in col][:10]
forex_sample = forex_cols[:5] if len(forex_cols) >= 5 else forex_cols

sample_cols = lme_sample + jpx_sample + us_sample + forex_sample
sample_data = train_df[sample_cols].dropna()

print(f"相関分析用サンプル列数: {len(sample_cols)}")
print(f"有効データ行数: {len(sample_data)}")

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
    plt.show()
    
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    print(f"\\n高相関ペア (|r| > 0.7): {len(corr_pairs)}")
    for col1, col2, corr in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"  {col1} - {col2}: {corr:.3f}")'''
    
    nb.cells.append(nbf.v4.new_code_cell(correlation_code))
    
    time_series_code = '''# Time series analysis
print("=== 時系列分析 ===")

if 'date_id' in train_df.columns:
    print(f"Date ID範囲: {train_df['date_id'].min()} - {train_df['date_id'].max()}")
    print(f"ユニークな日付: {train_df['date_id'].nunique()}")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    if lme_cols:
        for col in lme_cols:
            if col in train_df.columns:
                valid_data = train_df[['date_id', col]].dropna()
                if len(valid_data) > 0:
                    axes[0, 0].plot(valid_data['date_id'], valid_data[col], label=col, alpha=0.7)
        axes[0, 0].set_title('LME金属価格推移')
        axes[0, 0].set_xlabel('Date ID')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    jpx_close_cols = [col for col in jpx_cols if 'Close' in col][:3]
    for col in jpx_close_cols:
        if col in train_df.columns:
            valid_data = train_df[['date_id', col]].dropna()
            if len(valid_data) > 0:
                axes[0, 1].plot(valid_data['date_id'], valid_data[col], label=col, alpha=0.7)
    axes[0, 1].set_title('JPX先物価格推移 (サンプル)')
    axes[0, 1].set_xlabel('Date ID')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    us_close_cols = [col for col in us_cols if 'close' in col][:3]
    for col in us_close_cols:
        if col in train_df.columns:
            valid_data = train_df[['date_id', col]].dropna()
            if len(valid_data) > 0:
                axes[1, 0].plot(valid_data['date_id'], valid_data[col], label=col, alpha=0.7)
    axes[1, 0].set_title('US株式価格推移 (サンプル)')
    axes[1, 0].set_xlabel('Date ID')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for col in forex_cols[:3]:
        if col in train_df.columns:
            valid_data = train_df[['date_id', col]].dropna()
            if len(valid_data) > 0:
                axes[1, 1].plot(valid_data['date_id'], valid_data[col], label=col, alpha=0.7)
    axes[1, 1].set_title('外国為替レート推移 (サンプル)')
    axes[1, 1].set_xlabel('Date ID')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()'''
    
    nb.cells.append(nbf.v4.new_code_cell(time_series_code))
    
    data_quality_code = '''# Data quality assessment
print("=== データ品質評価 ===")

inf_counts = {}
for col in train_df.select_dtypes(include=[np.number]).columns:
    inf_count = np.isinf(train_df[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

print(f"無限値を含む列: {len(inf_counts)}")
if inf_counts:
    print("無限値の多い列 (上位10):")
    for col, count in sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {col}: {count}")

constant_cols = []
for col in train_df.select_dtypes(include=[np.number]).columns:
    if train_df[col].nunique() <= 1:
        constant_cols.append(col)

print(f"\\n定数列: {len(constant_cols)}")
if constant_cols:
    print(f"定数列例: {constant_cols[:10]}")

print("\\nデータ範囲分析 (数値列サンプル):")
numeric_sample = train_df.select_dtypes(include=[np.number]).columns[:10]
for col in numeric_sample:
    data = train_df[col].dropna()
    if len(data) > 0:
        print(f"  {col}: [{data.min():.6f}, {data.max():.6f}], std: {data.std():.6f}")

print(f"\\nメモリ使用量:")
print(f"  訓練データ: {train_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"  テストデータ: {test_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"  訓練ラベル: {train_labels.memory_usage(deep=True).sum() / 1024**2:.1f} MB")'''
    
    nb.cells.append(nbf.v4.new_code_cell(data_quality_code))
    
    summary_code = '''# Summary and insights
print("=== EDA要約と洞察 ===")

print("\\n主要な発見:")
print("1. データ構造:")
print(f"   - 訓練データ: {train_df.shape[0]:,}行 × {train_df.shape[1]}列")
print(f"   - {len(target_cols)}個のターゲット変数 (4つのラグ期間)")
print(f"   - 4つの主要市場: LME金属、JPX先物、US株式、外国為替")

print("\\n2. データ品質:")
missing_cols = sum(1 for col in train_df.columns if train_df[col].isnull().sum() > 0)
print(f"   - {missing_cols}/{len(train_df.columns)}列に欠損値")
print(f"   - 定数列: {len(constant_cols)}個")
print(f"   - 無限値を含む列: {len(inf_counts)}個")

print("\\n3. ターゲット変数:")
target_missing = train_labels[target_cols].isnull().sum()
print(f"   - 欠損値のないターゲット: {sum(target_missing == 0)}/{len(target_cols)}")
print(f"   - 平均値範囲: {np.mean(all_targets):.6f}")
print(f"   - 標準偏差: {np.std(all_targets):.6f}")

print("\\n4. 次のステップ:")
print("   - 特徴量エンジニアリング: 時系列特徴量、市場間関係")
print("   - 欠損値処理: 前方補完、移動平均")
print("   - モデル選択: 時系列対応、多ターゲット予測")
print("   - 評価指標: Sharpe ratio variant (Spearman correlation)")

print("\\nEDA完了 ✅")'''
    
    nb.cells.append(nbf.v4.new_code_cell(summary_code))
    
    with open('../notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("EDA notebook created successfully!")

if __name__ == "__main__":
    create_eda_notebook()
