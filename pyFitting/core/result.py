"""
pyFitting - Fit Result

This module defines the FitResult dataclass that stores all fitting results.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

from .types import ParameterSet


__all__ = ['FitResult']


@dataclass
class FitResult:
    """
    Container for fit results.
    
    This stores everything about a fit: data, parameters, fit values, metrics,
    and optimization information.
    
    Attributes:
    -----------
    data : IData
        Original data object
    parameters : ParameterSet
        Fitted parameters
    y_fit : np.ndarray
        Fitted y values
    metrics : Dict[str, float]
        Fit quality metrics (R², χ², etc.)
    parameter_errors : Dict[str, float]
        Parameter uncertainties
    covariance : Optional[np.ndarray]
        Covariance matrix
    optimizer : str
        Optimizer name
    success : bool
        Whether fit succeeded
    message : str
        Status message
    nfev : int
        Number of function evaluations
    nit : int
        Number of iterations
    metadata : Dict[str, Any]
        Additional metadata
    """
    
    # Core results
    data: 'IData'
    parameters: ParameterSet
    y_fit: np.ndarray
    
    # Quality metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Uncertainty
    parameter_errors: Dict[str, float] = field(default_factory=dict)
    covariance: Optional[np.ndarray] = None
    
    # Optimization info
    optimizer: str = ""
    success: bool = True
    message: str = ""
    nfev: int = 0
    nit: int = 0
    
    # Extra info
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self, verbose: bool = True):
        """
        Print summary of fit results.
        
        Parameters:
        -----------
        verbose : bool
            If True, print detailed information
        """
        print(f"\n{'='*70}")
        print(f"Fit Results ({self.optimizer})")
        print(f"{'='*70}")
        
        # Status
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        print(f"Status: {status}")
        print(f"Message: {self.message}")
        print(f"Function evaluations: {self.nfev}")
        print(f"Iterations: {self.nit}")
        
        # Parameters
        print(f"\n{'Parameters:':<30}")
        print(f"{'-'*70}")
        print(f"{'Name':<15} {'Value':>15} {'Error':>15} {'Bounds':>23}")
        print(f"{'-'*70}")
        
        for name, value in self.parameters.values.items():
            error = self.parameter_errors.get(name, 0.0)
            bounds = self.parameters.bounds.get(name, (None, None))
            fixed = self.parameters.fixed.get(name, False)
            
            bounds_str = f"[{bounds[0]:.2e}, {bounds[1]:.2e}]" if bounds[0] is not None else "[-, -]"
            fixed_str = " (fixed)" if fixed else ""
            
            print(f"{name:<15} {value:15.6e} {error:15.6e} {bounds_str:>23}{fixed_str}")
        
        # Metrics
        if self.metrics:
            print(f"\n{'Fit Quality Metrics:':<30}")
            print(f"{'-'*70}")
            for name, value in self.metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {name:<25}: {value:15.6f}")
                else:
                    print(f"  {name:<25}: {value}")
        
        print(f"{'='*70}\n")
        
        if verbose and self.covariance is not None:
            print("Covariance matrix available")
            print(f"Shape: {self.covariance.shape}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        
        Returns:
        --------
        Dict[str, Any] : result as dictionary
        """
        return {
            'parameters': self.parameters.values,
            'parameter_errors': self.parameter_errors,
            'metrics': self.metrics,
            'optimizer': self.optimizer,
            'success': self.success,
            'message': self.message,
            'nfev': self.nfev,
            'nit': self.nit,
            'metadata': self.metadata
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert result to pandas DataFrame (single row).
        
        Returns:
        --------
        pd.DataFrame : result as DataFrame
        """
        data = {}
        
        # Parameters
        for name, value in self.parameters.values.items():
            data[name] = value
            if name in self.parameter_errors:
                data[f"{name}_error"] = self.parameter_errors[name]
        
        # Metrics
        data.update(self.metrics)
        
        # Meta info
        data['optimizer'] = self.optimizer
        data['success'] = self.success
        data['nfev'] = self.nfev
        
        return pd.DataFrame([data])
    
    def get_residuals(self, relative: bool = False) -> np.ndarray:
        """
        Compute residuals.
        
        Parameters:
        -----------
        relative : bool
            If True, return relative residuals (residual/data)
        
        Returns:
        --------
        np.ndarray : residuals
        """
        y_data = self.data.get_y()
        residuals = y_data - self.y_fit
        
        if relative:
            residuals = residuals / np.maximum(np.abs(y_data), 1e-12)
        
        return residuals
    
    def get_relative_residuals_percent(self) -> np.ndarray:
        """Get relative residuals as percentage."""
        return self.get_residuals(relative=True) * 100
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        r2 = self.metrics.get('r2', 0.0)
        return f"FitResult({status}, R²={r2:.4f}, {len(self.parameters.values)} params)"