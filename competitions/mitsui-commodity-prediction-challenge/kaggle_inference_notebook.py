"""
The evaluation API requires that you set up a server which will respond to inference requests.
We have already defined the server; you just need write the predict function.

When we evaluate your submission on the hidden test set the client defined in `mitsui_gateway` will run in a different container
with direct access to the hidden test set and hand off the data timestep by timestep.

Your code will always have access to the published copies of the competition files.
"""

import os
import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server

NUM_TARGET_COLUMNS = 424

models = {}
feature_cols = None
train_median_values = None
models_loaded = False

def load_models():
    """Load and train models on first predict call"""
    global models, feature_cols, train_median_values, models_loaded
    
    if models_loaded:
        return
    
    print("Loading and training models...")
    
    train = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
    train_labels = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
    
    train = train.iloc[:-90]
    train_labels = train_labels.iloc[:-90]
    
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'date_id']
    
    features_df = train.copy()
    for col in numeric_cols[:50]:  # Limit features for speed
        if col in features_df.columns:
            features_df[f'{col}_log'] = np.log(features_df[col] + 1e-8)
            features_df[f'{col}_rolling_mean_5'] = features_df[col].rolling(5, min_periods=1).mean()
            features_df[f'{col}_rolling_std_5'] = features_df[col].rolling(5, min_periods=1).std()
    
    feature_cols = [col for col in features_df.columns if col != 'date_id']
    target_cols = [f'target_{i}' for i in range(NUM_TARGET_COLUMNS)]
    
    X_train = features_df[feature_cols]
    y_train = train_labels[target_cols]
    
    train_median_values = X_train.median()
    X_train = X_train.fillna(train_median_values)
    y_train = y_train.fillna(0)
    
    constant_cols = []
    for col in X_train.columns:
        if X_train[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        feature_cols = [col for col in feature_cols if col not in constant_cols]
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reduced for speed
        'learning_rate': 0.1,  # Increased for speed
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"Training {len(target_cols)} models...")
    
    for i, target_col in enumerate(target_cols):
        if i % 50 == 0:
            print(f"Training model {i+1}/{len(target_cols)}")
        
        y_target = y_train[target_col]
        
        train_data = lgb.Dataset(X_train, label=y_target)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=50,  # Reduced for speed
            callbacks=[lgb.log_evaluation(0)]
        )
        
        models[target_col] = model
    
    models_loaded = True
    print(f"✅ Loaded {len(models)} models")

def create_features(test_data: pd.DataFrame) -> pd.DataFrame:
    """Create features for test data matching training features"""
    features_df = test_data.copy()
    
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'date_id']
    
    for col in numeric_cols[:50]:  # Match training limit
        if col in features_df.columns:
            features_df[f'{col}_log'] = np.log(features_df[col] + 1e-8)
            features_df[f'{col}_rolling_mean_5'] = features_df[col].rolling(5, min_periods=1).mean()
            features_df[f'{col}_rolling_std_5'] = features_df[col].rolling(5, min_periods=1).std()
    
    return features_df

def predict(test: pl.DataFrame,
           label_lags_1_batch: pl.DataFrame,
           label_lags_2_batch: pl.DataFrame,
           label_lags_3_batch: pl.DataFrame,
           label_lags_4_batch: pl.DataFrame,
           ) -> pl.DataFrame | pd.DataFrame:
    """
    Replace this function with your inference code.
    
    You can return either a Pandas or Polars dataframe, though Polars is recommended for performance.
    Each batch of predictions (except the very first) must be returned within 1 minute of the batch features being provided.
    """
    global models, feature_cols, train_median_values, models_loaded
    
    if not models_loaded:
        load_models()
    
    test_pd = test.to_pandas() if isinstance(test, pl.DataFrame) else test
    
    # Create features
    features_df = create_features(test_pd)
    
    X_test = features_df[feature_cols]
    X_test = X_test.fillna(train_median_values)
    
    predictions = {}
    target_cols = [f'target_{i}' for i in range(NUM_TARGET_COLUMNS)]
    
    for target_col in target_cols:
        if target_col in models:
            pred = models[target_col].predict(X_test)
            predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
        else:
            predictions[target_col] = 0.0
    
    result_df = pd.DataFrame([predictions])
    
    assert isinstance(result_df, (pd.DataFrame, pl.DataFrame))
    assert len(result_df) == 1
    
    return result_df

inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))
