"""
pyFitting - Core Interfaces

This module defines the abstract interfaces that all components must implement.
Following the Dependency Inversion Principle, we depend on abstractions, not concrete classes.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Any
import numpy as np


__all__ = [
    'IData',
    'IModel', 
    'ILoss',
    'IOptimizer',
    'IEvaluator'
]


# ==============================================================================
# DATA INTERFACE
# ==============================================================================

class IData(ABC):
    """
    Abstract interface for data containers.
    
    Any data source (arrays, files, SAXS data, etc.) must implement this interface
    to work with the fitting framework.
    """
    
    @abstractmethod
    def get_x(self) -> np.ndarray:
        """
        Get x values (independent variable).
        
        Returns:
        --------
        np.ndarray : x values, 1D array
        """
        pass
    
    @abstractmethod
    def get_y(self) -> np.ndarray:
        """
        Get y values (dependent variable).
        
        Returns:
        --------
        np.ndarray : y values, 1D array
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> Optional[np.ndarray]:
        """
        Get weights for each data point (optional).
        
        Returns:
        --------
        Optional[np.ndarray] : weights, 1D array, or None
        """
        pass
    
    @abstractmethod
    def get_mask(self) -> np.ndarray:
        """
        Get mask indicating valid data points.
        
        Returns:
        --------
        np.ndarray : boolean mask, 1D array
        """
        pass
    
    @abstractmethod
    def transform(self, space: str) -> 'IData':
        """
        Transform data to different space (e.g., log, log-log).
        
        Parameters:
        -----------
        space : str
            Target space ('linear', 'log', 'log_log', etc.)
        
        Returns:
        --------
        IData : new data object in transformed space
        """
        pass
    
    @property
    @abstractmethod
    def space(self) -> str:
        """Current data space."""
        pass


# ==============================================================================
# MODEL INTERFACE
# ==============================================================================

class IModel(ABC):
    """
    Abstract interface for mathematical models.
    
    Any model (Gaussian, SAXS form factor, etc.) must implement this interface
    to work with the fitting framework.
    """
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Evaluate model at given x values with specified parameters.
        
        Parameters:
        -----------
        x : np.ndarray
            Independent variable values
        **params : dict
            Model parameters as keyword arguments
        
        Returns:
        --------
        np.ndarray : model values at x
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> 'ParameterSet':
        """
        Get parameter set (values, bounds, fixed/free).
        
        Returns:
        --------
        ParameterSet : parameter configuration
        """
        pass
    
    @abstractmethod
    def get_initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Get initial parameter guess from data.
        
        Parameters:
        -----------
        x : np.ndarray
            Independent variable
        y : np.ndarray
            Dependent variable
        
        Returns:
        --------
        Dict[str, float] : initial parameter values
        """
        pass
    
    def jacobian(self, x: np.ndarray, **params) -> Optional[np.ndarray]:
        """
        Compute Jacobian matrix (optional, for gradient-based optimizers).
        
        Parameters:
        -----------
        x : np.ndarray
            Independent variable
        **params : dict
            Model parameters
        
        Returns:
        --------
        Optional[np.ndarray] : Jacobian matrix, shape (len(x), n_params), or None
        """
        return None


# ==============================================================================
# LOSS INTERFACE
# ==============================================================================

class ILoss(ABC):
    """
    Abstract interface for loss functions.
    
    Any loss function (MSE, Chi2, correlation, etc.) must implement this interface.
    """
    
    @abstractmethod
    def compute(self, 
                y_data: np.ndarray, 
                y_model: np.ndarray,
                weights: Optional[np.ndarray] = None) -> float:
        """
        Compute loss between data and model.
        
        Parameters:
        -----------
        y_data : np.ndarray
            Observed data values
        y_model : np.ndarray
            Model predicted values
        weights : Optional[np.ndarray]
            Weights for each point
        
        Returns:
        --------
        float : loss value (lower is better)
        """
        pass
    
    def gradient(self,
                 y_data: np.ndarray,
                 y_model: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute gradient of loss w.r.t. model values (optional).
        
        Parameters:
        -----------
        y_data : np.ndarray
            Observed data values
        y_model : np.ndarray
            Model predicted values
        weights : Optional[np.ndarray]
            Weights for each point
        
        Returns:
        --------
        Optional[np.ndarray] : gradient vector, or None
        """
        return None


# ==============================================================================
# OPTIMIZER INTERFACE
# ==============================================================================

class IOptimizer(ABC):
    """
    Abstract interface for optimization algorithms.
    
    Any optimizer (L-BFGS-B, SLSQP, DE, etc.) must implement this interface.
    """
    
    @abstractmethod
    def optimize(self,
                 objective: callable,
                 x0: np.ndarray,
                 bounds: list,
                 **options) -> 'OptimizeResult':
        """
        Run optimization to minimize objective function.
        
        Parameters:
        -----------
        objective : callable
            Function to minimize, signature: f(x) -> float
        x0 : np.ndarray
            Initial parameter guess
        bounds : list of tuples
            Parameter bounds [(low, high), ...]
        **options : dict
            Optimizer-specific options
        
        Returns:
        --------
        OptimizeResult : optimization result
        """
        pass


# ==============================================================================
# EVALUATOR INTERFACE
# ==============================================================================

class IEvaluator(ABC):
    """
    Abstract interface for fit quality evaluation.
    
    Any evaluator (standard metrics, uncertainty, diagnostics) must implement this.
    """
    
    @abstractmethod
    def evaluate(self, result: 'FitResult') -> Dict[str, Any]:
        """
        Evaluate fit quality and compute metrics.
        
        Parameters:
        -----------
        result : FitResult
            Fit result to evaluate
        
        Returns:
        --------
        Dict[str, Any] : computed metrics
        """
        pass