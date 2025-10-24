"""
pyFitting.models.base - Base Model Class

This module provides BaseModel, the base class for all models.
"""

import numpy as np
from typing import Dict, Optional
from abc import abstractmethod

from pyFitting.core.interfaces import IModel
from pyFitting.core.types import ParameterSet


__all__ = ['BaseModel']


class BaseModel(IModel):
    """
    Base class for all models.
    
    Subclasses must implement:
    - evaluate(x, **params)
    - get_initial_guess(x, y)
    
    This class provides common functionality like parameter management.
    
    Examples:
    ---------
    >>> class MyModel(BaseModel):
    ...     def evaluate(self, x, a, b):
    ...         return a * x + b
    ...     
    ...     def get_initial_guess(self, x, y):
    ...         return {'a': 1.0, 'b': 0.0}
    >>> 
    >>> model = MyModel()
    >>> model.set_parameters(a=2.0, b=1.0)
    >>> y = model.evaluate(x, **model.get_parameters().values)
    """
    
    def __init__(self):
        """Initialize base model."""
        self._parameters = ParameterSet()
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Evaluate model (must be implemented by subclass).
        
        Parameters:
        -----------
        x : np.ndarray
            Independent variable
        **params : dict
            Model parameters
        
        Returns:
        --------
        np.ndarray : model values
        """
        pass
    
    @abstractmethod
    def get_initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Get initial parameter guess (must be implemented by subclass).
        
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
    
    def get_parameters(self) -> ParameterSet:
        """Get parameter set."""
        return self._parameters
    
    def set_parameters(self, **params):
        """
        Set parameter values.
        
        Parameters:
        -----------
        **params : dict
            Parameter values as keyword arguments
        """
        self._parameters.values.update(params)
    
    def set_bounds(self, **bounds):
        """
        Set parameter bounds.
        
        Parameters:
        -----------
        **bounds : dict
            Parameter bounds as keyword arguments, each value is (low, high)
        
        Examples:
        ---------
        >>> model.set_bounds(a=(0, 10), b=(-5, 5))
        """
        self._parameters.bounds.update(bounds)
    
    def fix_parameter(self, name: str, value: Optional[float] = None):
        """
        Fix a parameter at current or specified value.
        
        Parameters:
        -----------
        name : str
            Parameter name
        value : Optional[float]
            Value to fix at (uses current if None)
        """
        self._parameters.fix_parameter(name, value)
    
    def free_parameter(self, name: str):
        """
        Free a previously fixed parameter.
        
        Parameters:
        -----------
        name : str
            Parameter name
        """
        self._parameters.free_parameter(name)
    
    def __call__(self, x: np.ndarray, **params) -> np.ndarray:
        """
        Convenience method to evaluate model.
        
        Parameters:
        -----------
        x : np.ndarray
            Independent variable
        **params : dict
            Model parameters
        
        Returns:
        --------
        np.ndarray : model values
        """
        return self.evaluate(x, **params)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self._parameters.values)} params)"