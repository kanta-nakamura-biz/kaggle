import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
import os


class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_train_data(self):
        return pd.read_csv(self.config['data']['train_path'])
    
    def load_test_data(self):
        return pd.read_csv(self.config['data']['test_path'])
    
    def load_data(self):
        train = self.load_train_data()
        test = self.load_test_data()
        return train, test


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        
    def handle_missing_values(self, df, strategy='mean'):
        df_processed = df.copy()
        
        if strategy == 'mean':
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
                df_processed[numeric_columns].mean()
            )
        elif strategy == 'median':
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
                df_processed[numeric_columns].median()
            )
        elif strategy == 'mode':
            for col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        elif strategy == 'drop':
            df_processed = df_processed.dropna()
            
        return df_processed
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        if method == 'none':
            return X_train, X_test
            
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['feature_scaler'] = scaler
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def encode_categorical(self, df, method='onehot'):
        df_encoded = df.copy()
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns)
        elif method == 'label':
            for col in categorical_columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                
        return df_encoded
    
    def save_preprocessors(self, path):
        os.makedirs(path, exist_ok=True)
        
        if self.scalers:
            joblib.dump(self.scalers, os.path.join(path, 'scalers.joblib'))
        if self.encoders:
            joblib.dump(self.encoders, os.path.join(path, 'encoders.joblib'))
    
    def load_preprocessors(self, path):
        scaler_path = os.path.join(path, 'scalers.joblib')
        encoder_path = os.path.join(path, 'encoders.joblib')
        
        if os.path.exists(scaler_path):
            self.scalers = joblib.load(scaler_path)
        if os.path.exists(encoder_path):
            self.encoders = joblib.load(encoder_path)


def create_cv_folds(df, config):
    cv_strategy = config['training']['cv_strategy']
    n_folds = config['training']['cv_folds']
    target_col = config['data']['target_column']
    
    if cv_strategy == 'kfold':
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=config['experiment']['seed'])
        folds = list(cv.split(df))
    elif cv_strategy == 'stratified':
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['experiment']['seed'])
        folds = list(cv.split(df, df[target_col]))
    elif cv_strategy == 'group':
        group_col = config['data'].get('group_column', 'group')
        cv = GroupKFold(n_splits=n_folds)
        folds = list(cv.split(df, df[target_col], df[group_col]))
    elif cv_strategy == 'time_series':
        cv = TimeSeriesSplit(n_splits=n_folds)
        folds = list(cv.split(df))
    else:
        raise ValueError(f"Unknown CV strategy: {cv_strategy}")
    
    return folds


def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
          f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df
