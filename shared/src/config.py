import yaml
import os
from pathlib import Path


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self._find_config_file()
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _find_config_file(self):
        possible_paths = [
            'config.yaml',
            'configs/config.yaml',
            '../configs/config.yaml',
            '../../configs/config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Config file not found. Please specify config_path.")
    
    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path=None):
        if path is None:
            path = self.config_path
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def update_from_dict(self, update_dict):
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, update_dict)
    
    def create_experiment_config(self, experiment_name, base_config=None):
        if base_config is None:
            base_config = self.config.copy()
        
        base_config['experiment']['name'] = experiment_name
        
        experiment_dir = f"experiments/{experiment_name}"
        os.makedirs(experiment_dir, exist_ok=True)
        
        config_path = f"{experiment_dir}/config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)
        
        return Config(config_path)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def __contains__(self, key):
        return key in self.config


def load_config(config_path=None):
    return Config(config_path)


def create_default_config():
    default_config = {
        'experiment': {
            'name': 'default',
            'description': 'Default experiment',
            'seed': 42
        },
        'data': {
            'train_path': 'data/raw/train.csv',
            'test_path': 'data/raw/test.csv',
            'target_column': 'target',
            'id_column': 'id'
        },
        'preprocessing': {
            'missing_strategy': 'mean',
            'scaling': 'standard',
            'encoding': 'onehot'
        },
        'model': {
            'name': 'lightgbm',
            'params': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        },
        'training': {
            'cv_folds': 5,
            'cv_strategy': 'kfold',
            'early_stopping_rounds': 100,
            'num_boost_round': 10000
        },
        'evaluation': {
            'metrics': ['rmse', 'mae', 'r2']
        },
        'output': {
            'model_dir': 'models/',
            'submission_dir': 'submissions/',
            'log_dir': 'logs/'
        }
    }
    
    return default_config


def validate_config(config):
    required_keys = [
        'experiment.name',
        'data.train_path',
        'data.target_column',
        'model.name',
        'training.cv_folds'
    ]
    
    for key in required_keys:
        if config.get(key) is None:
            raise ValueError(f"Required config key '{key}' is missing")
    
    return True
