import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append('../../shared/src')
from data_utils import DataLoader as BaseDataLoader


class MitsuiDataLoader(BaseDataLoader):
    """
    Custom data loader for MITSUI Commodity Prediction Challenge
    """
    
    def __init__(self, data_dir: str = "data/raw/"):
        super().__init__()
        self.data_dir = Path(data_dir)
    
    def load_competition_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all competition data files
        """
        data = {}
        
        expected_files = [
            'train.csv',
            'test.csv', 
            'train_labels.csv',
            'target_pairs.csv'
        ]
        
        for file_name in expected_files:
            file_path = self.data_dir / file_name
            if file_path.exists():
                print(f"Loading {file_name}...")
                data[file_name.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"Shape: {data[file_name.replace('.csv', '')].shape}")
            else:
                print(f"Warning: {file_name} not found in {self.data_dir}")
        
        lagged_dir = self.data_dir / "lagged_test_labels"
        if lagged_dir.exists():
            data['lagged_test_labels'] = {}
            for lag_file in lagged_dir.glob("test_labels_lag_*.csv"):
                lag_num = lag_file.stem.split('_')[-1]
                data['lagged_test_labels'][f'lag_{lag_num}'] = pd.read_csv(lag_file)
                print(f"Loaded {lag_file.name}, shape: {data['lagged_test_labels'][f'lag_{lag_num}'].shape}")
        
        return data
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Get summary statistics for all loaded data
        """
        summary = {}
        
        for name, df in data.items():
            if isinstance(df, pd.DataFrame):
                summary[name] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                }
        
        return summary
    
    def validate_data_structure(self, data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Validate the loaded data structure and return any issues
        """
        issues = []
        
        required_files = ['train', 'test', 'train_labels']
        for file_name in required_files:
            if file_name not in data:
                issues.append(f"Missing required file: {file_name}.csv")
        
        if 'train' in data and 'test' in data:
            train_cols = set(data['train'].columns)
            test_cols = set(data['test'].columns)
            
            if not train_cols.intersection(test_cols):
                issues.append("No common columns found between train and test sets")
        
        return issues
