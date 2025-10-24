"""
pyFitting.evaluation.metrics - Fit Quality Metrics

This module provides evaluators for computing fit quality metrics.
"""

import numpy as np
from typing import Dict, Any
from scipy.stats import pearsonr

from pyFitting.core.interfaces import IEvaluator
from pyFitting.core.result import FitResult


__all__ = ['StandardEvaluator']


class StandardEvaluator(IEvaluator):
    """
    Standard fit quality metrics evaluator.
    
    Computes:
    - R² (coefficient of determination)
    - Pearson correlation
    - Chi-squared and reduced chi-squared
    - RMSE (root mean squared error)
    - MAE (mean absolute error)
    
    Examples:
    ---------
    >>> evaluator = StandardEvaluator()
    >>> metrics = evaluator.evaluate(result)
    >>> print(metrics['r2'], metrics['chi2_reduced'])
    """
    
    def evaluate(self, result: FitResult) -> Dict[str, Any]:
        """
        Compute standard fit quality metrics.
        
        Parameters:
        -----------
        result : FitResult
            Fit result to evaluate
        
        Returns:
        --------
        Dict[str, Any] : computed metrics
        """
        y_data = result.data.get_y()
        y_fit = result.y_fit
        weights = result.data.get_weights()
        n_params = len(result.parameters.get_free_params())
        
        metrics = {}
        
        # Basic residuals
        residuals = y_data - y_fit
        
        # R² (coefficient of determination)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        metrics['r2'] = 1 - ss_res / (ss_tot + 1e-30)
        
        # R² in log space
        log_y_data = np.log(np.maximum(y_data, 1e-12))
        log_y_fit = np.log(np.maximum(y_fit, 1e-12))
        log_residuals = log_y_data - log_y_fit
        ss_res_log = np.sum(log_residuals ** 2)
        ss_tot_log = np.sum((log_y_data - np.mean(log_y_data)) ** 2)
        metrics['r2_log'] = 1 - ss_res_log / (ss_tot_log + 1e-30)
        
        # Pearson correlation
        try:
            corr, p_value = pearsonr(y_data, y_fit)
            metrics['pearson_r'] = corr
            metrics['p_value'] = p_value
        except:
            metrics['pearson_r'] = 0.0
            metrics['p_value'] = 1.0
        
        # Correlation in log space
        try:
            corr_log, _ = pearsonr(log_y_data, log_y_fit)
            metrics['pearson_r_log'] = corr_log
        except:
            metrics['pearson_r_log'] = 0.0
        
        # Chi-squared
        if weights is not None:
            chi2 = np.sum((residuals**2) * weights)
        else:
            # Use Poisson-like errors
            sigma = np.sqrt(np.maximum(y_data, 1.0))
            chi2 = np.sum((residuals / sigma) ** 2)
        
        dof = len(y_data) - n_params
        metrics['chi2'] = chi2
        metrics['chi2_reduced'] = chi2 / dof if dof > 0 else chi2
        metrics['dof'] = dof
        
        # RMSE (root mean squared error)
        metrics['rmse'] = np.sqrt(np.mean(residuals ** 2))
        metrics['rmse_log'] = np.sqrt(np.mean(log_residuals ** 2))
        
        # MAE (mean absolute error)
        metrics['mae'] = np.mean(np.abs(residuals))
        
        # Max absolute error
        metrics['max_error'] = np.max(np.abs(residuals))
        
        # Mean relative error
        rel_error = np.abs(residuals) / np.maximum(np.abs(y_data), 1e-12)
        metrics['mean_rel_error'] = np.mean(rel_error)
        metrics['max_rel_error'] = np.max(rel_error)
        
        return metrics
    
    def __repr__(self) -> str:
        return "StandardEvaluator()"