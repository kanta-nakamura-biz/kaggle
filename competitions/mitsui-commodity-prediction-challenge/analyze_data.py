import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append('../../shared/src')

from data.data_loader import MitsuiDataLoader

def main():
    print("=== MITSUI Commodity Prediction Challenge - Data Analysis ===\n")
    
    loader = MitsuiDataLoader('data/raw/')
    data = loader.load_competition_data()
    
    summary = loader.get_data_summary(data)
    for name, info in summary.items():
        print(f'{name.upper()}:')
        print(f'  Shape: {info["shape"]}')
        print(f'  Columns: {len(info["columns"])} columns')
        print(f'  Memory: {info["memory_usage"]/1024/1024:.1f} MB')
        if info['missing_values']:
            missing = {k:v for k,v in info['missing_values'].items() if v > 0}
            if missing:
                print(f'  Missing values: {len(missing)} columns with missing data')
        print()
    
    issues = loader.validate_data_structure(data)
    if issues:
        print(f'DATA ISSUES: {issues}\n')
    else:
        print('Data structure validation: PASSED\n')
    
    if 'train' in data:
        train_df = data['train']
        print("TRAIN DATA COLUMNS:")
        print(f"Total columns: {len(train_df.columns)}")
        
        lme_cols = [col for col in train_df.columns if 'LME_' in col]
        jpx_cols = [col for col in train_df.columns if 'JPX_' in col]
        us_cols = [col for col in train_df.columns if 'US_' in col]
        forex_cols = [col for col in train_df.columns if any(fx in col for fx in ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD'])]
        
        print(f"  LME columns: {len(lme_cols)}")
        print(f"  JPX columns: {len(jpx_cols)}")
        print(f"  US Stock columns: {len(us_cols)}")
        print(f"  Forex columns: {len(forex_cols)}")
        print(f"  Other columns: {len(train_df.columns) - len(lme_cols) - len(jpx_cols) - len(us_cols) - len(forex_cols)}")
        
        print(f"\nSample LME columns: {lme_cols[:5]}")
        print(f"Sample JPX columns: {jpx_cols[:5]}")
        print(f"Sample US columns: {us_cols[:5]}")
        
    if 'target_pairs' in data:
        target_pairs = data['target_pairs']
        print(f"\nTARGET PAIRS:")
        print(f"Shape: {target_pairs.shape}")
        print("First 10 target pairs:")
        print(target_pairs.head(10))
        
        if 'lag' in target_pairs.columns:
            lag_counts = target_pairs['lag'].value_counts().sort_index()
            print(f"\nLag distribution:")
            print(lag_counts)
    
    if 'train_labels' in data:
        train_labels = data['train_labels']
        print(f"\nTRAIN LABELS:")
        print(f"Shape: {train_labels.shape}")
        print(f"Columns: {list(train_labels.columns)}")
        
        target_cols = [col for col in train_labels.columns if col.startswith('target_')]
        if target_cols:
            print(f"\nTarget columns found: {len(target_cols)}")
            print("Statistics for first 5 targets:")
            print(train_labels[target_cols[:5]].describe())

if __name__ == "__main__":
    main()
