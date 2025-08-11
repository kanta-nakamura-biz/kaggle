import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append('../../shared/src')

from data.data_loader import MitsuiDataLoader
from features.feature_engineering import FinancialFeatureEngineer

def main():
    print("=== MITSUI Commodity Prediction - Feature Engineering ===")
    
    loader = MitsuiDataLoader('data/raw/')
    data = loader.load_competition_data()
    
    train_df = data['train']
    test_df = data['test']
    train_labels = data['train_labels']
    target_pairs = data['target_pairs']
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")
    
    feature_engineer = FinancialFeatureEngineer()
    
    print("\n=== Creating features for training data ===")
    train_features = feature_engineer.create_features(train_df)
    print(f"Train features shape: {train_features.shape}")
    print(f"New features created: {train_features.shape[1] - train_df.shape[1]}")
    
    print("\n=== Creating features for test data ===")
    test_features = feature_engineer.create_features(test_df)
    print(f"Test features shape: {test_features.shape}")
    
    print("\n=== Feature Analysis by Market ===")
    
    feature_counts = {
        'original': 0,
        'rolling': 0,
        'volatility': 0,
        'momentum': 0,
        'cross_market': 0,
        'technical': 0
    }
    
    for col in train_features.columns:
        if any(x in col for x in ['_rolling_', '_ma_', '_ewm_']):
            feature_counts['rolling'] += 1
        elif any(x in col for x in ['_volatility_', '_std_']):
            feature_counts['volatility'] += 1
        elif any(x in col for x in ['_momentum_', '_roc_', '_rsi_']):
            feature_counts['momentum'] += 1
        elif any(x in col for x in ['_cross_', '_ratio_', '_diff_']):
            feature_counts['cross_market'] += 1
        elif any(x in col for x in ['_sma_', '_ema_', '_bb_']):
            feature_counts['technical'] += 1
        else:
            feature_counts['original'] += 1
    
    print("Feature counts by type:")
    for feat_type, count in feature_counts.items():
        print(f"  {feat_type}: {count}")
    
    print("\n=== Saving processed features ===")
    os.makedirs('data/processed', exist_ok=True)
    
    train_features.to_csv('data/processed/train_features.csv', index=False)
    test_features.to_csv('data/processed/test_features.csv', index=False)
    train_labels.to_csv('data/processed/train_labels.csv', index=False)
    
    print("✅ Features saved to data/processed/")
    
    print("\n=== Feature Quality Analysis ===")
    
    train_missing = train_features.isnull().sum()
    missing_features = train_missing[train_missing > 0]
    print(f"Features with missing values: {len(missing_features)}/{len(train_features.columns)}")
    
    constant_features = []
    for col in train_features.select_dtypes(include=[np.number]).columns:
        if train_features[col].nunique() <= 1:
            constant_features.append(col)
    
    print(f"Constant features: {len(constant_features)}")
    
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        sample_cols = numeric_cols[:100]  # Sample for correlation analysis
        corr_matrix = train_features[sample_cols].corr()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.95:  # Very high correlation
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        print(f"High correlation pairs (|r| > 0.95): {len(high_corr_pairs)}")
    
    print("\n=== Feature Engineering Complete ===")
    print("Next steps:")
    print("1. Model training with engineered features")
    print("2. Cross-validation and hyperparameter tuning")
    print("3. Inference and submission")

if __name__ == "__main__":
    main()
