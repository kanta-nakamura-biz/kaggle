import os
import pandas as pd
import numpy as np
import polars as pl
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.mitsui_inference_server

NUM_TARGET_COLUMNS = 424

class FinancialFeatureEngineer:
    """
    Feature engineering class adapted for real-time inference
    """
    
    def __init__(self, windows: List[int] = [5, 10, 20]):
        self.windows = windows
        self.feature_names = []
    
    def create_features_realtime(self, test_data: pd.DataFrame, lag_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create features for real-time inference using current test data and lag batches
        """
        features_df = test_data.copy()
        
        numeric_cols = test_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_id']
        
        features_df = self._add_basic_features(features_df, numeric_cols)
        
        features_df = self._add_lag_based_features(features_df, lag_data, numeric_cols)
        
        features_df = self._add_volatility_features_realtime(features_df, lag_data, numeric_cols)
        
        features_df = self._add_momentum_features_realtime(features_df, lag_data, numeric_cols)
        
        features_df = self._add_cross_market_features_realtime(features_df)
        
        return features_df
    
    def _add_basic_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Add basic features that don't require historical data"""
        for col in numeric_cols[:30]:
            if col in df.columns:
                df[f'{col}_log'] = np.log(df[col] + 1e-8)
                df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        
        return df
    
    def _add_lag_based_features(self, df: pd.DataFrame, lag_data: Dict[str, pd.DataFrame], numeric_cols: List[str]) -> pd.DataFrame:
        """Add features based on lag data from previous timesteps"""
        for lag_key, lag_df in lag_data.items():
            if lag_df is not None and not lag_df.empty:
                for col in numeric_cols[:20]:
                    if col in lag_df.columns:
                        lag_values = lag_df[col].values
                        if len(lag_values) > 0:
                            df[f'{col}_{lag_key}_mean'] = np.mean(lag_values)
                            df[f'{col}_{lag_key}_std'] = np.std(lag_values)
                            df[f'{col}_{lag_key}_last'] = lag_values[-1] if len(lag_values) > 0 else 0
                            
                            if len(lag_values) >= 2:
                                df[f'{col}_{lag_key}_diff'] = lag_values[-1] - lag_values[-2]
                                df[f'{col}_{lag_key}_pct_change'] = (lag_values[-1] - lag_values[-2]) / (lag_values[-2] + 1e-8)
        
        return df
    
    def _add_volatility_features_realtime(self, df: pd.DataFrame, lag_data: Dict[str, pd.DataFrame], numeric_cols: List[str]) -> pd.DataFrame:
        """Add volatility features using lag data"""
        for lag_key, lag_df in lag_data.items():
            if lag_df is not None and not lag_df.empty:
                for col in numeric_cols[:15]:
                    if col in lag_df.columns:
                        lag_values = lag_df[col].values
                        if len(lag_values) >= 3:
                            pct_changes = np.diff(lag_values) / (lag_values[:-1] + 1e-8)
                            df[f'{col}_{lag_key}_volatility'] = np.std(pct_changes)
                            df[f'{col}_{lag_key}_realized_vol'] = np.mean(pct_changes ** 2)
        
        return df
    
    def _add_momentum_features_realtime(self, df: pd.DataFrame, lag_data: Dict[str, pd.DataFrame], numeric_cols: List[str]) -> pd.DataFrame:
        """Add momentum features using lag data"""
        for lag_key, lag_df in lag_data.items():
            if lag_df is not None and not lag_df.empty:
                for col in numeric_cols[:15]:
                    if col in lag_df.columns:
                        lag_values = lag_df[col].values
                        if len(lag_values) >= 5:
                            roc_5 = (lag_values[-1] - lag_values[-5]) / (lag_values[-5] + 1e-8)
                            df[f'{col}_{lag_key}_roc_5'] = roc_5
                            
                            if len(lag_values) >= 10:
                                sma_short = np.mean(lag_values[-5:])
                                sma_long = np.mean(lag_values[-10:])
                                df[f'{col}_{lag_key}_sma_ratio'] = sma_short / (sma_long + 1e-8)
        
        return df
    
    def _add_cross_market_features_realtime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-market features using current data"""
        lme_cols = [col for col in df.columns if 'LME_' in col and 'Close' in col]
        jpx_cols = [col for col in df.columns if 'JPX_' in col and 'Close' in col]
        us_cols = [col for col in df.columns if 'US_Stock_' in col and 'close' in col]
        
        if len(lme_cols) >= 2:
            for i, col1 in enumerate(lme_cols[:2]):
                for col2 in lme_cols[i+1:3]:
                    if col1 in df.columns and col2 in df.columns:
                        df[f'{col1}_to_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
        
        return df

class MitsuiPredictor:
    """
    Main predictor class for MITSUI competition inference server
    """
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = FinancialFeatureEngineer()
        self.feature_cols = None
        self.target_cols = [f'target_{i}' for i in range(NUM_TARGET_COLUMNS)]
        self.models_loaded = False
        self.train_median_values = None
    
    def load_models(self):
        """Load pre-trained LightGBM models"""
        print("Loading models...")
        
        train = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train.csv')
        train_labels = pd.read_csv('/kaggle/input/mitsui-commodity-prediction-challenge/train_labels.csv')
        
        train = train.iloc[:-90]
        train_labels = train_labels.iloc[:-90]
        
        sample_size = min(100, len(train))
        train_sample = train.iloc[-sample_size:]
        
        lag_data = {}
        if len(train) > sample_size + 10:
            lag_data['lag_1'] = train.iloc[-(sample_size+10):-sample_size]
        if len(train) > sample_size + 20:
            lag_data['lag_2'] = train.iloc[-(sample_size+20):-(sample_size+10)]
        
        train_features = self.feature_engineer.create_features_realtime(train_sample, lag_data)
        
        self.feature_cols = [col for col in train_features.columns if col != 'date_id']
        
        X_train = train_features[self.feature_cols]
        self.train_median_values = X_train.median()
        
        X_train = X_train.fillna(self.train_median_values)
        y_train_sample = train_labels.iloc[-sample_size:][self.target_cols].fillna(0)
        
        constant_cols = []
        for col in X_train.columns:
            if X_train[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            X_train = X_train.drop(columns=constant_cols)
            self.feature_cols = [col for col in self.feature_cols if col not in constant_cols]
        
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
        
        print(f"Training {len(self.target_cols)} models...")
        
        for i, target_col in enumerate(self.target_cols):
            if i % 50 == 0:
                print(f"Training model {i+1}/{len(self.target_cols)}")
            
            y_target = y_train_sample[target_col]
            
            train_data = lgb.Dataset(X_train, label=y_target)
            
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            self.models[target_col] = model
        
        self.models_loaded = True
        print(f"✅ Loaded {len(self.models)} models")
    
    def predict_batch(self, test: pd.DataFrame, lag_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Make predictions for a single batch"""
        if not self.models_loaded:
            self.load_models()
        
        features_df = self.feature_engineer.create_features_realtime(test, lag_data)
        
        X_test = features_df[self.feature_cols]
        X_test = X_test.fillna(self.train_median_values)
        
        predictions = {}
        
        for target_col in self.target_cols:
            if target_col in self.models:
                pred = self.models[target_col].predict(X_test)
                predictions[target_col] = pred[0] if len(pred) > 0 else 0.0
            else:
                predictions[target_col] = 0.0
        
        result_df = pd.DataFrame([predictions])
        
        return result_df

predictor = MitsuiPredictor()

def predict(test: pl.DataFrame,
           label_lags_1_batch: pl.DataFrame,
           label_lags_2_batch: pl.DataFrame,
           label_lags_3_batch: pl.DataFrame,
           label_lags_4_batch: pl.DataFrame,
           ) -> pl.DataFrame | pd.DataFrame:
    """
    Main prediction function for MITSUI competition evaluation API
    """
    
    test_pd = test.to_pandas() if isinstance(test, pl.DataFrame) else test
    
    lag_data = {}
    
    if label_lags_1_batch is not None:
        lag_data['lag_1'] = label_lags_1_batch.to_pandas() if isinstance(label_lags_1_batch, pl.DataFrame) else label_lags_1_batch
    
    if label_lags_2_batch is not None:
        lag_data['lag_2'] = label_lags_2_batch.to_pandas() if isinstance(label_lags_2_batch, pl.DataFrame) else label_lags_2_batch
    
    if label_lags_3_batch is not None:
        lag_data['lag_3'] = label_lags_3_batch.to_pandas() if isinstance(label_lags_3_batch, pl.DataFrame) else label_lags_3_batch
    
    if label_lags_4_batch is not None:
        lag_data['lag_4'] = label_lags_4_batch.to_pandas() if isinstance(label_lags_4_batch, pl.DataFrame) else label_lags_4_batch
    
    predictions_df = predictor.predict_batch(test_pd, lag_data)
    
    assert len(predictions_df) == 1, f"Expected 1 prediction row, got {len(predictions_df)}"
    assert len(predictions_df.columns) == NUM_TARGET_COLUMNS, f"Expected {NUM_TARGET_COLUMNS} columns, got {len(predictions_df.columns)}"
    
    return predictions_df

# Instantiate the inference server
inference_server = kaggle_evaluation.mitsui_inference_server.MitsuiInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('/kaggle/input/mitsui-commodity-prediction-challenge/',))
