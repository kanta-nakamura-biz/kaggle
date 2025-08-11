import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append('../../shared/src')

from data.data_loader import MitsuiDataLoader
from evaluation import sharpe_ratio_spearman

def main():
    print("=== MITSUI Commodity Prediction - Model Training ===")
    
    print("Loading processed features...")
    train_features = pd.read_csv('data/processed/train_features.csv')
    test_features = pd.read_csv('data/processed/test_features.csv')
    train_labels = pd.read_csv('data/processed/train_labels.csv')
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    
    feature_cols = [col for col in train_features.columns if col != 'date_id']
    target_cols = [col for col in train_labels.columns if col.startswith('target_')]
    
    X_train = train_features[feature_cols]
    X_test = test_features[feature_cols]
    y_train = train_labels[target_cols]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns: {len(target_cols)}")
    
    print("Handling missing values...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test
    y_train = y_train.fillna(0)  # Fill target missing values with 0
    
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
    
    print("Starting time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = []
    models = {}
    
    sample_targets = target_cols[:10]
    print(f"Training models for {len(sample_targets)} targets (sample)")
    
    for i, target_col in enumerate(sample_targets):
        print(f"Training model for {target_col} ({i+1}/{len(sample_targets)})")
        
        y_target = y_train[target_col]
        
        target_cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_target.iloc[train_idx], y_target.iloc[val_idx]
            
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            target_cv_scores.append(rmse)
        
        avg_cv_score = np.mean(target_cv_scores)
        cv_scores.append(avg_cv_score)
        print(f"  CV RMSE: {avg_cv_score:.6f}")
        
        train_data = lgb.Dataset(X_train, label=y_target)
        final_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        models[target_col] = final_model
    
    print(f"\nOverall CV RMSE: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    
    print("Generating predictions...")
    predictions = {}
    
    for target_col in sample_targets:
        model = models[target_col]
        pred = model.predict(X_test)
        predictions[target_col] = pred
    
    print("Creating submission format...")
    submission_data = []
    
    for i, target_col in enumerate(sample_targets):
        for j, pred_value in enumerate(predictions[target_col]):
            submission_data.append({
                'row_id': f"{j}_{target_col}",
                'target': pred_value
            })
    
    submission_df = pd.DataFrame(submission_data)
    
    print("Saving models and predictions...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    for target_col, model in models.items():
        model.save_model(f'models/lgb_model_{target_col}.txt')
    
    submission_df.to_csv('submissions/sample_submission.csv', index=False)
    
    print("Analyzing feature importance...")
    if sample_targets:
        first_model = models[sample_targets[0]]
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': first_model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        print("Top 20 most important features:")
        print(feature_importance.head(20))
        
        feature_importance.to_csv('models/feature_importance.csv', index=False)
    
    print("\n=== Model Training Complete ===")
    print(f"Models saved: {len(models)}")
    print(f"Submission shape: {submission_df.shape}")
    print("Next steps:")
    print("1. Train models for all 424 targets")
    print("2. Generate full test predictions")
    print("3. Submit to Kaggle competition")

if __name__ == "__main__":
    main()
