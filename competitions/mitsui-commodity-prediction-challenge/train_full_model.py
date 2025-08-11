import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append('../../shared/src')

from data.data_loader import MitsuiDataLoader
from evaluation import sharpe_ratio_spearman

def main():
    print("=== MITSUI Commodity Prediction - Full Model Training ===")
    
    print("Loading processed features...")
    train_features = pd.read_csv('data/processed/train_features.csv')
    test_features = pd.read_csv('data/processed/test_features.csv')
    train_labels = pd.read_csv('data/processed/train_labels.csv')
    
    loader = MitsuiDataLoader('data/raw/')
    data = loader.load_competition_data()
    target_pairs = data['target_pairs']
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Target pairs shape: {target_pairs.shape}")
    
    feature_cols = [col for col in train_features.columns if col != 'date_id']
    target_cols = [col for col in train_labels.columns if col.startswith('target_')]
    
    X_train = train_features[feature_cols]
    X_test = test_features[feature_cols]
    y_train = train_labels[target_cols]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns: {len(target_cols)}")
    
    print("Handling missing values...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    y_train = y_train.fillna(0)
    
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns")
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)
    
    print(f"Final feature shape: {X_train.shape}")
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"Training models for all {len(target_cols)} targets...")
    
    models = {}
    cv_scores = []
    predictions = {}
    
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for efficiency
    
    for i, target_col in enumerate(target_cols):
        if i % 50 == 0:
            print(f"Training model for {target_col} ({i+1}/{len(target_cols)})")
        
        y_target = y_train[target_col]
        
        train_idx, val_idx = list(tscv.split(X_train))[-1]  # Use last split
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_target.iloc[train_idx], y_target.iloc[val_idx]
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,  # Reduced for efficiency
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        cv_scores.append(rmse)
        
        train_data_full = lgb.Dataset(X_train, label=y_target)
        final_model = lgb.train(
            lgb_params,
            train_data_full,
            num_boost_round=model.best_iteration,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        models[target_col] = final_model
        
        pred = final_model.predict(X_test)
        predictions[target_col] = pred
    
    print(f"\nOverall CV RMSE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    
    print("Creating submission file...")
    submission_data = []
    
    target_mapping = {}
    for idx, row in target_pairs.iterrows():
        target_mapping[f"target_{idx}"] = {
            'symbol_1': row['symbol_1'],
            'symbol_2': row['symbol_2'],
            'lag': row['lag']
        }
    
    for test_idx in range(len(X_test)):
        for target_idx, target_col in enumerate(target_cols):
            if target_col in predictions:
                row_id = f"{test_idx}_{target_idx}"
                target_value = predictions[target_col][test_idx]
                submission_data.append({
                    'row_id': row_id,
                    'target': target_value
                })
    
    submission_df = pd.DataFrame(submission_data)
    
    print(f"Submission shape: {submission_df.shape}")
    print("Sample submission:")
    print(submission_df.head(10))
    
    print("Saving models and submission...")
    os.makedirs('models/full', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    submission_df.to_csv('submissions/submission.csv', index=False)
    
    model_summary = {
        'num_models': len(models),
        'cv_rmse_mean': np.mean(cv_scores),
        'cv_rmse_std': np.std(cv_scores),
        'num_features': X_train.shape[1],
        'submission_rows': len(submission_df)
    }
    
    import json
    with open('models/full/model_summary.json', 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    if target_cols:
        first_model = models[target_cols[0]]
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': first_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('models/full/feature_importance.csv', index=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
    
    print("\n=== Full Model Training Complete ===")
    print(f"✅ Trained {len(models)} models")
    print(f"✅ Generated {len(submission_df)} predictions")
    print(f"✅ CV RMSE: {np.mean(cv_scores):.6f}")
    print(f"✅ Submission saved: submissions/submission.csv")
    
    return submission_df

if __name__ == "__main__":
    submission = main()
