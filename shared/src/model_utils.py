import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import os
import optuna


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.models = {}
        
    def get_model(self, model_name, params):
        if model_name == 'lightgbm':
            if params.get('objective') in ['regression', 'rmse', 'mae']:
                return lgb.LGBMRegressor(**params)
            else:
                return lgb.LGBMClassifier(**params)
        elif model_name == 'xgboost':
            if params.get('objective') in ['reg:squarederror', 'reg:absoluteerror']:
                return xgb.XGBRegressor(**params)
            else:
                return xgb.XGBClassifier(**params)
        elif model_name == 'catboost':
            if params.get('objective') in ['RMSE', 'MAE']:
                return cb.CatBoostRegressor(**params, verbose=False)
            else:
                return cb.CatBoostClassifier(**params, verbose=False)
        elif model_name == 'random_forest':
            if params.get('task_type', 'regression') == 'regression':
                return RandomForestRegressor(**params)
            else:
                return RandomForestClassifier(**params)
        elif model_name == 'linear':
            if params.get('task_type', 'regression') == 'regression':
                return LinearRegression(**params)
            else:
                return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_cv(self, X, y, folds):
        model_name = self.config['model']['name']
        model_params = self.config['model']['params']
        
        oof_predictions = np.zeros(len(X))
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"Training fold {fold_idx + 1}/{len(folds)}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.get_model(model_name, model_params)
            
            if model_name in ['lightgbm', 'xgboost', 'catboost']:
                if hasattr(model, 'fit'):
                    if model_name == 'lightgbm':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(self.config['training']['early_stopping_rounds'])]
                        )
                    elif model_name == 'xgboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=self.config['training']['early_stopping_rounds'],
                            verbose=False
                        )
                    elif model_name == 'catboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_val, y_val),
                            early_stopping_rounds=self.config['training']['early_stopping_rounds'],
                            verbose=False
                        )
            else:
                model.fit(X_train, y_train)
            
            val_pred = model.predict(X_val)
            oof_predictions[val_idx] = val_pred
            
            fold_score = self.calculate_metric(y_val, val_pred, self.config['evaluation']['metrics'][0])
            cv_scores.append(fold_score)
            print(f"Fold {fold_idx + 1} score: {fold_score:.6f}")
            
            self.models[f'fold_{fold_idx}'] = model
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        print(f"CV Score: {cv_mean:.6f} ± {cv_std:.6f}")
        
        return oof_predictions, cv_scores
    
    def predict_test(self, X_test):
        predictions = []
        
        for fold_idx, model in self.models.items():
            pred = model.predict(X_test)
            predictions.append(pred)
        
        return np.mean(predictions, axis=0)
    
    def calculate_metric(self, y_true, y_pred, metric):
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        elif metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            return f1_score(y_true, y_pred, average='weighted')
        elif metric == 'auc':
            return roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def save_models(self, path):
        os.makedirs(path, exist_ok=True)
        
        for fold_name, model in self.models.items():
            model_path = os.path.join(path, f'{fold_name}_model.joblib')
            joblib.dump(model, model_path)
    
    def load_models(self, path):
        self.models = {}
        
        for filename in os.listdir(path):
            if filename.endswith('_model.joblib'):
                fold_name = filename.replace('_model.joblib', '')
                model_path = os.path.join(path, filename)
                self.models[fold_name] = joblib.load(model_path)


class HyperparameterOptimizer:
    def __init__(self, config):
        self.config = config
        
    def objective(self, trial):
        model_name = self.config['model']['name']
        
        if model_name == 'lightgbm':
            params = {
                'objective': self.config['model']['params']['objective'],
                'metric': self.config['model']['params']['metric'],
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1
            }
        elif model_name == 'xgboost':
            params = {
                'objective': self.config['model']['params']['objective'],
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': self.config['experiment']['seed']
            }
        else:
            raise ValueError(f"Hyperparameter optimization not implemented for {model_name}")
        
        temp_config = self.config.copy()
        temp_config['model']['params'] = params
        
        trainer = ModelTrainer(temp_config)
        
        return 0.0
    
    def optimize(self, X, y, folds):
        study = optuna.create_study(
            direction=self.config['optuna']['direction'],
            sampler=getattr(optuna.samplers, f"{self.config['optuna']['sampler'].upper()}Sampler")()
        )
        
        study.optimize(self.objective, n_trials=self.config['optuna']['n_trials'])
        
        return study.best_params, study.best_value
