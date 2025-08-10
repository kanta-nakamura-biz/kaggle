import logging
import os
import sys
from datetime import datetime
from pathlib import Path


class KaggleLogger:
    def __init__(self, name='kaggle', log_dir='logs', level=logging.INFO):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'{self.name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)
    
    def log_experiment_start(self, config):
        self.info("=" * 50)
        self.info(f"実験開始: {config.get('experiment.name', 'Unknown')}")
        self.info(f"説明: {config.get('experiment.description', 'No description')}")
        self.info(f"シード: {config.get('experiment.seed', 'Not set')}")
        self.info(f"モデル: {config.get('model.name', 'Unknown')}")
        self.info("=" * 50)
    
    def log_data_info(self, train_shape, test_shape=None):
        self.info(f"訓練データ形状: {train_shape}")
        if test_shape:
            self.info(f"テストデータ形状: {test_shape}")
    
    def log_cv_results(self, cv_scores, metric_name='Score'):
        mean_score = sum(cv_scores) / len(cv_scores)
        std_score = (sum((x - mean_score) ** 2 for x in cv_scores) / len(cv_scores)) ** 0.5
        
        self.info(f"クロスバリデーション結果:")
        for i, score in enumerate(cv_scores):
            self.info(f"  Fold {i+1}: {score:.6f}")
        self.info(f"  平均 {metric_name}: {mean_score:.6f} ± {std_score:.6f}")
    
    def log_model_performance(self, metrics_dict):
        self.info("モデル性能:")
        for metric, value in metrics_dict.items():
            self.info(f"  {metric}: {value:.6f}")
    
    def log_feature_importance(self, feature_importance_df, top_n=10):
        self.info(f"上位{top_n}特徴量重要度:")
        for idx, row in feature_importance_df.head(top_n).iterrows():
            self.info(f"  {row['feature']}: {row['importance']:.6f}")
    
    def log_submission_info(self, submission_path, score=None):
        self.info(f"提出ファイル作成: {submission_path}")
        if score:
            self.info(f"予想スコア: {score:.6f}")
    
    def log_experiment_end(self):
        self.info("=" * 50)
        self.info("実験終了")
        self.info("=" * 50)


def setup_logging(name='kaggle', log_dir='logs', level=logging.INFO):
    return KaggleLogger(name, log_dir, level)


def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('kaggle')
        logger.debug(f"関数呼び出し: {func.__name__}")
        logger.debug(f"引数: args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"関数完了: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"関数エラー: {func.__name__} - {str(e)}")
            raise
    
    return wrapper


class ExperimentTracker:
    def __init__(self, experiment_name, log_dir='logs'):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.experiment_log = []
        
        self.logger = KaggleLogger(f'experiment_{experiment_name}', log_dir)
    
    def log_step(self, step_name, details=None):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'step': step_name,
            'details': details or {}
        }
        
        self.experiment_log.append(log_entry)
        self.logger.info(f"ステップ: {step_name}")
        
        if details:
            for key, value in details.items():
                self.logger.info(f"  {key}: {value}")
    
    def save_experiment_log(self):
        import json
        
        log_file = self.log_dir / f'experiment_{self.experiment_name}_log.json'
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"実験ログ保存: {log_file}")
    
    def get_experiment_summary(self):
        return {
            'experiment_name': self.experiment_name,
            'total_steps': len(self.experiment_log),
            'start_time': self.experiment_log[0]['timestamp'] if self.experiment_log else None,
            'end_time': self.experiment_log[-1]['timestamp'] if self.experiment_log else None,
            'steps': [entry['step'] for entry in self.experiment_log]
        }
