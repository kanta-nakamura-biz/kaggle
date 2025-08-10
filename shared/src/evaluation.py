import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_regression(self, y_true, y_pred, prefix=''):
        metrics = {}
        
        metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
        metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
        
        residuals = y_true - y_pred
        metrics[f'{prefix}mean_residual'] = np.mean(residuals)
        metrics[f'{prefix}std_residual'] = np.std(residuals)
        
        return metrics
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None, prefix=''):
        metrics = {}
        
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics[f'{prefix}recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, average='weighted')
        
        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:
                metrics[f'{prefix}auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics[f'{prefix}auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        
        return metrics
    
    def plot_regression_results(self, y_true, y_pred, title='Regression Results'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('True vs Predicted')
        
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predictions')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        axes[1, 0].hist(residuals, bins=30, alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_classification_results(self, y_true, y_pred, y_pred_proba=None, class_names=None):
        n_plots = 2 if y_pred_proba is None else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        else:
            axes = [axes[0], axes[1], axes[2]]
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        if class_names:
            axes[0].set_xticklabels(class_names)
            axes[0].set_yticklabels(class_names)
        
        metrics = self.evaluate_classification(y_true, y_pred, y_pred_proba)
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        axes[1].bar(metric_names, metric_values)
        axes[1].set_title('Classification Metrics')
        axes[1].set_ylabel('Score')
        axes[1].set_ylim(0, 1)
        
        for i, v in enumerate(metric_values):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = metrics.get('auc', 0)
            
            axes[2].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[2].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[2].set_xlabel('False Positive Rate')
            axes[2].set_ylabel('True Positive Rate')
            axes[2].set_title('ROC Curve')
            axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def feature_importance_plot(self, model, feature_names, top_n=20):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            print("Model doesn't have feature importance information")
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def cross_validation_plot(self, cv_scores, title='Cross Validation Scores'):
        plt.figure(figsize=(10, 6))
        
        folds = range(1, len(cv_scores) + 1)
        plt.plot(folds, cv_scores, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(cv_scores):.4f}')
        plt.axhline(y=np.mean(cv_scores) + np.std(cv_scores), color='r', 
                   linestyle=':', alpha=0.7, label=f'±1 Std: {np.std(cv_scores):.4f}')
        plt.axhline(y=np.mean(cv_scores) - np.std(cv_scores), color='r', 
                   linestyle=':', alpha=0.7)
        
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def learning_curve_plot(self, train_scores, val_scores, title='Learning Curve'):
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_scores) + 1)
        plt.plot(epochs, train_scores, 'b-', label='Training Score', linewidth=2)
        plt.plot(epochs, val_scores, 'r-', label='Validation Score', linewidth=2)
        
        plt.xlabel('Epoch/Iteration')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()


def calculate_competition_metric(y_true, y_pred, metric_name):
    if metric_name.lower() == 'rmse':
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric_name.lower() == 'mae':
        return mean_absolute_error(y_true, y_pred)
    elif metric_name.lower() == 'rmsle':
        return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    elif metric_name.lower() == 'mape':
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    elif metric_name.lower() == 'r2':
        return r2_score(y_true, y_pred)
    elif metric_name.lower() == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric_name.lower() == 'f1':
        return f1_score(y_true, y_pred, average='weighted')
    elif metric_name.lower() == 'auc':
        return roc_auc_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
