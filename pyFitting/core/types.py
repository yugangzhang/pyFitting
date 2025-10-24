"""
pyFitting - Core Types

This module defines enums, dataclasses, and common types used throughout the framework.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import numpy as np


__all__ = [
    'FitSpace',
    'OptimizerType',
    'ParameterSet',
    'OptimizeResult'
]


# ==============================================================================
# ENUMS
# ==============================================================================

class FitSpace(Enum):
    """Available data transformation spaces"""
    LINEAR = "linear"
    LOG = "log"
    LOG_LOG = "log_log"


class OptimizerType(Enum):
    """Optimizer categories"""
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"


# ==============================================================================
# PARAMETER MANAGEMENT
# ==============================================================================

@dataclass
class ParameterSet:
    """
    Container for model parameters with bounds and fixed/free status.
    
    Attributes:
    -----------
    values : Dict[str, float]
        Parameter values
    bounds : Dict[str, Tuple[float, float]]
        Parameter bounds (lower, upper)
    fixed : Dict[str, bool]
        Whether parameter is fixed (True) or free (False)
    """
    
    values: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fixed: Dict[str, bool] = field(default_factory=dict)
    
    def get_free_params(self) -> Dict[str, float]:
        """
        Get only the free (non-fixed) parameters.
        
        Returns:
        --------
        Dict[str, float] : free parameters
        """
        return {k: v for k, v in self.values.items() 
                if not self.fixed.get(k, False)}
    
    def get_free_names(self) -> list:
        """Get names of free parameters in order."""
        return list(self.get_free_params().keys())
    
    def to_array(self) -> np.ndarray:
        """
        Convert free parameters to numpy array for optimization.
        
        Returns:
        --------
        np.ndarray : free parameter values
        """
        return np.array([v for k, v in self.get_free_params().items()])
    
    def from_array(self, arr: np.ndarray) -> None:
        """
        Update free parameters from numpy array.
        
        Parameters:
        -----------
        arr : np.ndarray
            New parameter values
        """
        free_names = self.get_free_names()
        if len(arr) != len(free_names):
            raise ValueError(f"Expected {len(free_names)} values, got {len(arr)}")
        
        for name, value in zip(free_names, arr):
            self.values[name] = float(value)
    
    def get_bounds_array(self) -> list:
        """
        Get bounds for free parameters as list of tuples.
        
        Returns:
        --------
        list : [(low, high), ...] for each free parameter
        """
        return [self.bounds.get(name, (-np.inf, np.inf)) 
                for name in self.get_free_names()]
    
    def set_parameter(self, name: str, value: float, 
                     bounds: Optional[Tuple[float, float]] = None,
                     fixed: bool = False):
        """
        Set a single parameter.
        
        Parameters:
        -----------
        name : str
            Parameter name
        value : float
            Parameter value
        bounds : Optional[Tuple[float, float]]
            Parameter bounds (low, high)
        fixed : bool
            Whether parameter is fixed
        """
        self.values[name] = float(value)
        if bounds is not None:
            self.bounds[name] = bounds
        self.fixed[name] = fixed
    
    def fix_parameter(self, name: str, value: Optional[float] = None):
        """
        Fix a parameter at current or specified value.
        
        Parameters:
        -----------
        name : str
            Parameter name
        value : Optional[float]
            Value to fix at (uses current value if None)
        """
        if value is not None:
            self.values[name] = float(value)
        self.fixed[name] = True
    
    def free_parameter(self, name: str):
        """
        Free a previously fixed parameter.
        
        Parameters:
        -----------
        name : str
            Parameter name
        """
        self.fixed[name] = False
    
    def copy(self) -> 'ParameterSet':
        """Create a deep copy of this parameter set."""
        return ParameterSet(
            values=self.values.copy(),
            bounds=self.bounds.copy(),
            fixed=self.fixed.copy()
        )
    
    def __repr__(self) -> str:
        free = self.get_free_names()
        fixed = [k for k, v in self.fixed.items() if v]
        return f"ParameterSet({len(self.values)} params, {len(free)} free, {len(fixed)} fixed)"


# ==============================================================================
# OPTIMIZATION RESULT
# ==============================================================================

@dataclass
class OptimizeResult:
    """
    Result from optimization.
    
    Attributes:
    -----------
    x : np.ndarray
        Optimized parameter values
    fun : float
        Final objective function value
    success : bool
        Whether optimization succeeded
    message : str
        Status message
    nfev : int
        Number of function evaluations
    nit : int
        Number of iterations
    """
    
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nfev: int = 0
    nit: int = 0
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"OptimizeResult({status}, fun={self.fun:.4e}, nfev={self.nfev})"