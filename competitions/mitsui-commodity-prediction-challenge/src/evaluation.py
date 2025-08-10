import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sharpe_ratio_spearman(y_true, y_pred):
    """
    Calculate Sharpe ratio variant using Spearman rank correlation
    as used in the MITSUI Commodity Prediction Challenge
    """
    correlation, _ = spearmanr(y_true, y_pred)
    
    if np.isnan(correlation):
        return 0.0
    
    return correlation


def calculate_sharpe_variant(returns_true, returns_pred):
    """
    Calculate the Sharpe ratio variant for commodity returns prediction
    """
    if len(returns_true) != len(returns_pred):
        raise ValueError("True and predicted returns must have the same length")
    
    correlation = sharpe_ratio_spearman(returns_true, returns_pred)
    
    return correlation


def evaluate_predictions(y_true, y_pred):
    """
    Comprehensive evaluation including the competition metric
    """
    results = {}
    
    results['sharpe_spearman'] = sharpe_ratio_spearman(y_true, y_pred)
    
    results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
    results['mae'] = np.mean(np.abs(y_true - y_pred))
    
    correlation_pearson = np.corrcoef(y_true, y_pred)[0, 1]
    results['correlation_pearson'] = correlation_pearson if not np.isnan(correlation_pearson) else 0.0
    
    return results
