"""
pyFitting.optimizers.local - Local Optimization Methods

This module provides local optimization algorithms using scipy.optimize.
"""

from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
#from scipy.optimize import minimize
from scipy.optimize import  minimize, differential_evolution, basinhopping, dual_annealing, shgo

from pyFitting.core.interfaces import IOptimizer
from pyFitting.core.types import OptimizeResult


__all__ = ['LocalOptimizer']

"""
pyFitting.optimizers.global - Global Optimization Methods
"""

class LocalOptimizer(IOptimizer):
    """
    Local optimization using scipy.optimize.minimize.
    
    Supports various methods: SLSQP, L-BFGS-B, Powell, TNC, trust-constr, etc.
    
    Parameters:
    -----------
    method : str
        Optimization method (see scipy.optimize.minimize)
        Recommended: 'SLSQP' (robust, good for most problems)
    
    Examples:
    ---------
    >>> optimizer = LocalOptimizer('SLSQP')
    >>> result = optimizer.optimize(objective, x0, bounds)
    """
    
    # Available methods
    METHODS = [
        'SLSQP',        # Sequential Least Squares (recommended)
        'L-BFGS-B',     # L-BFGS with bounds (fast but less robust)
        'TNC',          # Truncated Newton
        'trust-constr', # Trust-region (most robust)
        'Powell',       # Powell's method (derivative-free)
        'Nelder-Mead',  # Simplex method (derivative-free)
        'COBYLA'        # Constrained optimization
    ]
    
    def __init__(self, method: str = 'SLSQP'):
        """
        Initialize local optimizer.
        
        Parameters:
        -----------
        method : str
            Optimization method
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {self.METHODS}")
        
        self.method = method
    
    def optimize(self,
                 objective: Callable,
                 x0: np.ndarray,
                 bounds: List[Tuple[float, float]],
                 **options) -> OptimizeResult:
        """
        Run local optimization.
        
        Parameters:
        -----------
        objective : callable
            Objective function to minimize, signature: f(x) -> float
        x0 : np.ndarray
            Initial parameter guess
        bounds : list of tuples
            Parameter bounds [(low, high), ...]
        **options : dict
            Additional options for scipy.optimize.minimize
        
        Returns:
        --------
        OptimizeResult : optimization result
        """
        # Set default options
        default_options = {
            'maxiter': 1000,
            'ftol': 1e-9,
            'disp': False
        }
        default_options.update(options)
        
        # Run optimization
        result = minimize(
            objective,
            x0,
            method=self.method,
            bounds=bounds,
            options=default_options
        )
        
        # Convert to our result type
        return OptimizeResult(
            x=result.x,
            fun=float(result.fun),
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            nit=int(getattr(result, 'nit', 0))
        )
    
    def __repr__(self) -> str:
        return f"LocalOptimizer(method='{self.method}')"


def compare_optimizers(objective: Callable,
                      x0: np.ndarray,
                      bounds: List[Tuple[float, float]],
                      methods: Optional[List[str]] = None,
                      **options) -> Dict[str, OptimizeResult]:
    """
    Compare multiple optimization methods.
    
    Parameters:
    -----------
    objective : callable
        Objective function
    x0 : np.ndarray
        Initial guess
    bounds : list of tuples
        Parameter bounds
    methods : list of str, optional
        Methods to compare (default: ['SLSQP', 'L-BFGS-B', 'Powell'])
    **options : dict
        Options passed to optimizers
    
    Returns:
    --------
    Dict[str, OptimizeResult] : results for each method
    """
    if methods is None:
        methods = ['SLSQP', 'trust-constr', 'Powell']
    
    results = {}
    for method in methods:
        try:
            optimizer = LocalOptimizer(method)
            result = optimizer.optimize(objective, x0, bounds, **options)
            results[method] = result
        except Exception as e:
            print(f"Method {method} failed: {e}")
    
    return results