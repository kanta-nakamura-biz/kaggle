import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FinancialFeatureEngineer:
    """
    Feature engineering class for financial time series data
    Specialized for commodity prediction with multi-market data
    """
    
    def __init__(self, windows: List[int] = [5, 10, 20, 50]):
        self.windows = windows
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for financial time series
        """
        print("Creating financial time series features...")
        
        features_df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'date_id']
        
        print(f"Processing {len(numeric_cols)} numeric columns...")
        
        features_df = self._add_rolling_features(features_df, numeric_cols)
        
        features_df = self._add_volatility_features(features_df, numeric_cols)
        
        features_df = self._add_momentum_features(features_df, numeric_cols)
        
        features_df = self._add_technical_indicators(features_df, numeric_cols)
        
        features_df = self._add_cross_market_features(features_df)
        
        features_df = self._add_lag_features(features_df, numeric_cols)
        
        print(f"Feature engineering complete. Shape: {features_df.shape}")
        return features_df
    
    def _add_rolling_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Add rolling window statistics"""
        print("Adding rolling features...")
        
        for window in self.windows:
            for col in numeric_cols[:50]:  # Limit to first 50 columns for efficiency
                if col in df.columns:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    
                    if window <= 20:  # Only for smaller windows
                        df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                        df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Add volatility-based features"""
        print("Adding volatility features...")
        
        for col in numeric_cols[:30]:  # Limit for efficiency
            if col in df.columns:
                df[f'{col}_pct_change'] = df[col].pct_change()
                df[f'{col}_diff'] = df[col].diff()
                
                for window in [5, 10, 20]:
                    df[f'{col}_volatility_{window}'] = df[f'{col}_pct_change'].rolling(window=window, min_periods=1).std()
                    
                    df[f'{col}_realized_vol_{window}'] = (df[f'{col}_pct_change'] ** 2).rolling(window=window, min_periods=1).mean()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Add momentum and trend features"""
        print("Adding momentum features...")
        
        for col in numeric_cols[:30]:  # Limit for efficiency
            if col in df.columns:
                for period in [5, 10, 20]:
                    df[f'{col}_roc_{period}'] = df[col].pct_change(periods=period)
                
                delta = df[col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-8)
                df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
                
                ema_12 = df[col].ewm(span=12).mean()
                ema_26 = df[col].ewm(span=26).mean()
                df[f'{col}_macd'] = ema_12 - ema_26
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Add technical analysis indicators"""
        print("Adding technical indicators...")
        
        for col in numeric_cols[:20]:  # Limit for efficiency
            if col in df.columns:
                for window in [5, 10, 20]:
                    sma = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_sma_{window}'] = sma
                    df[f'{col}_price_to_sma_{window}'] = df[col] / (sma + 1e-8)
                
                for span in [5, 10, 20]:
                    ema = df[col].ewm(span=span).mean()
                    df[f'{col}_ema_{span}'] = ema
                    df[f'{col}_price_to_ema_{span}'] = df[col] / (ema + 1e-8)
                
                sma_20 = df[col].rolling(window=20, min_periods=1).mean()
                std_20 = df[col].rolling(window=20, min_periods=1).std()
                df[f'{col}_bb_upper'] = sma_20 + (2 * std_20)
                df[f'{col}_bb_lower'] = sma_20 - (2 * std_20)
                df[f'{col}_bb_position'] = (df[col] - df[f'{col}_bb_lower']) / (df[f'{col}_bb_upper'] - df[f'{col}_bb_lower'] + 1e-8)
        
        return df
    
    def _add_cross_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-market relationship features"""
        print("Adding cross-market features...")
        
        lme_cols = [col for col in df.columns if 'LME_' in col and 'Close' in col]
        jpx_cols = [col for col in df.columns if 'JPX_' in col and 'Close' in col]
        us_cols = [col for col in df.columns if 'US_Stock_' in col and 'close' in col]
        forex_cols = [col for col in df.columns if any(fx in col for fx in ['USD', 'EUR', 'GBP', 'JPY'])]
        
        if len(lme_cols) >= 2:
            for i, col1 in enumerate(lme_cols[:2]):
                for col2 in lme_cols[i+1:3]:
                    if col1 in df.columns and col2 in df.columns:
                        df[f'{col1}_to_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
        
        if len(jpx_cols) >= 1 and len(us_cols) >= 1:
            jpx_sample = jpx_cols[0]
            us_sample = us_cols[0]
            if jpx_sample in df.columns and us_sample in df.columns:
                window = 20
                jpx_returns = df[jpx_sample].pct_change()
                us_returns = df[us_sample].pct_change()
                df[f'jpx_us_correlation_{window}'] = jpx_returns.rolling(window=window, min_periods=1).corr(us_returns)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Add lagged features"""
        print("Adding lag features...")
        
        key_cols = [col for col in numeric_cols if any(x in col for x in ['LME_', 'JPX_']) and 'Close' in col]
        
        for col in key_cols[:10]:  # Limit for efficiency
            if col in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by type for analysis
        """
        return {
            'original': [name for name in self.feature_names if not any(x in name for x in ['_rolling_', '_volatility_', '_momentum_', '_cross_', '_lag_'])],
            'rolling': [name for name in self.feature_names if '_rolling_' in name],
            'volatility': [name for name in self.feature_names if '_volatility_' in name or '_pct_change' in name],
            'momentum': [name for name in self.feature_names if any(x in name for x in ['_roc_', '_rsi', '_macd'])],
            'technical': [name for name in self.feature_names if any(x in name for x in ['_sma_', '_ema_', '_bb_'])],
            'cross_market': [name for name in self.feature_names if '_cross_' in name or '_ratio' in name or '_correlation' in name],
            'lag': [name for name in self.feature_names if '_lag_' in name]
        }
