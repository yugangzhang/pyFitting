"""
pyFitting.loss.standard - Standard Loss Functions

This module provides commonly used loss functions.
"""

import numpy as np
from typing import Optional

from pyFitting.core.interfaces import ILoss
from pyFitting.utils.common import (
get_ab_correlation, 
get_similarity_by_overlap,
 
)

MIN = 1e-10 #1e-18

__all__ = [
    'OverLapLoss',
    'MSELoss',
    'Chi2Loss',
    'CorrelationLoss',
    'HybridLoss'
]


class OverlapLoss(ILoss):
    """
    Overlap loss (negative overlap for minimization).
    
    Overlap = I12 / (I11 + I22 - |I12|)

    where   I12 = dot(y_data, y_model)
            I11 = dot(y_data, y_data)
            I11 = dot(y_data, y_data)
    
    Loss = -Overlap (we minimize, so negate to maximize overlap)
    Parameters:
    -----------
    use_log : bool
        If True, compute MSE in log space
    """
    
    def __init__(self, use_log: bool = False):
        self.use_log = use_log
    
    def compute(self, y_data: np.ndarray, y_model: np.ndarray,
                weights: Optional[np.ndarray] = None) -> float:
        if self.use_log:
            y_data = np.log(np.maximum(y_data, MIN))
            y_model = np.log(np.maximum(y_model, MIN))
        #get_similarity_by_overlap(y_data, y_fit)  
 
        residuals = get_similarity_by_overlap(y_data, y_model )      
        
        if weights is not None:
            residuals = residuals * weights
            return -float(np.mean(residuals))
        
        return  -residuals 
    
    def __repr__(self) -> str:
        space = "log" if self.use_log else "linear"
        return f"OverLapLoss(space='{space}')"
        

class MSELoss(ILoss):
    """
    Mean Squared Error loss.
    
    Loss = mean((y_data - y_model)^2)
    
    Parameters:
    -----------
    use_log : bool
        If True, compute MSE in log space
    """
    
    def __init__(self, use_log: bool = False):
        self.use_log = use_log
    
    def compute(self, y_data: np.ndarray, y_model: np.ndarray,
                weights: Optional[np.ndarray] = None) -> float:
        if self.use_log:
            y_data = np.log(np.maximum(y_data, MIN))
            y_model = np.log(np.maximum(y_model, MIN))
        
        residuals = (y_data - y_model) ** 2
        
        if weights is not None:
            residuals = residuals * weights
        
        return float(np.mean(residuals))
    
    def __repr__(self) -> str:
        space = "log" if self.use_log else "linear"
        return f"MSELoss(space='{space}')"


class Chi2Loss(ILoss):
    """
    Chi-squared loss.
    
    Loss = sum(((y_data - y_model) / sigma)^2)
    
    Parameters:
    -----------
    use_log : bool
        If True, compute chi2 in log space
    """
    
    def __init__(self, use_log: bool = False):
        self.use_log = use_log
    
    def compute(self, y_data: np.ndarray, y_model: np.ndarray,
                weights: Optional[np.ndarray] = None) -> float:
        if self.use_log:
            y_data = np.log(np.maximum(y_data, MIN))
            y_model = np.log(np.maximum(y_model, MIN))
        
        if weights is None:
            # Use Poisson-like errors
            sigma = np.sqrt(np.maximum(np.abs(y_data), 1.0))
        else:
            # weights = 1/sigma^2, so sigma = 1/sqrt(weights)
            sigma = 1.0 / np.sqrt(np.maximum(weights, MIN))
        
        residuals = (y_data - y_model) / sigma
        return float(np.sum(residuals ** 2))
    
    def __repr__(self) -> str:
        space = "log" if self.use_log else "linear"
        return f"Chi2Loss(space='{space}')"


class CorrelationLoss(ILoss):
    """
    Negative correlation loss.
    
    Loss = -corr(y_data, y_model)
    
    Minimizing this maximizes correlation.
    
    Parameters:
    -----------
    use_log : bool
        If True, compute correlation in log space
    """
    
    def __init__(self, use_log: bool = True):
        self.use_log = use_log
    
    def compute(self, y_data: np.ndarray, y_model: np.ndarray,
                weights: Optional[np.ndarray] = None) -> float:
        if self.use_log:
            y_data = np.log(np.maximum(y_data, MIN))
            y_model = np.log(np.maximum(y_model, MIN))
        
        if len(y_data) < 2:
            return 1.0
        
        # Compute weighted correlation if weights provided
        if weights is not None:
            w = weights / (weights.sum() + 1e-30)
            mean_data = np.sum(w * y_data)
            mean_model = np.sum(w * y_model)
            num = np.sum(w * (y_data - mean_data) * (y_model - mean_model))
            den = np.sqrt(np.sum(w * (y_data - mean_data)**2) * 
                         np.sum(w * (y_model - mean_model)**2))
        else:
            mean_data = np.mean(y_data)
            mean_model = np.mean(y_model)
            num = np.sum((y_data - mean_data) * (y_model - mean_model))
            den = np.sqrt(np.sum((y_data - mean_data)**2) * 
                         np.sum((y_model - mean_model)**2))
        
        if den < 1e-30:
            return 1.0
        
        corr = num / den
        return -float(corr)  # Negative because we minimize
    
    def __repr__(self) -> str:
        space = "log" if self.use_log else "linear"
        return f"CorrelationLoss(space='{space}')"


class HybridLoss(ILoss):
    """
    Hybrid loss combining correlation and MSE.
    
    Loss = alpha * correlation_loss + (1-alpha) * normalized_mse_loss
    
    Parameters:
    -----------
    alpha : float
        Weight for correlation term (0 to 1)
    use_log : bool
        If True, compute in log space
    """
    
    def __init__(self, alpha: float = 0.7, use_log: bool = True):
        self.alpha = alpha
        self.use_log = use_log
        self._corr_loss = CorrelationLoss(use_log=use_log)
        self._mse_loss = MSELoss(use_log=use_log)
    
    def compute(self, y_data: np.ndarray, y_model: np.ndarray,
                weights: Optional[np.ndarray] = None) -> float:
        corr_loss = self._corr_loss.compute(y_data, y_model, weights)
        mse_loss = self._mse_loss.compute(y_data, y_model, weights)
        
        # Normalize MSE
        if self.use_log:
            y_trans = np.log(np.maximum(y_data, MIN))
        else:
            y_trans = y_data
        
        mse_normalized = mse_loss / (np.mean(y_trans**2) + 1e-30)
        
        return self.alpha * corr_loss + (1 - self.alpha) * mse_normalized
    
    def __repr__(self) -> str:
        space = "log" if self.use_log else "linear"
        return f"HybridLoss(alpha={self.alpha}, space='{space}')"