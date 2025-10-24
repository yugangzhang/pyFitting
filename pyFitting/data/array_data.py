"""
pyFitting.data.array_data - Simple Array-Based Data

This module provides ArrayData, the simplest data container for x, y arrays.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from pyFitting.core.interfaces import IData


__all__ = ['ArrayData']


@dataclass
class ArrayData(IData):
    """
    Simple array-based data container.
    
    This is the most basic data type - just x and y arrays with optional weights.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    weights : array-like, optional
        Weights for each point
    mask : array-like, optional
        Boolean mask for valid points
    space : str
        Data space ('linear', 'log', 'log_log')
    
    Examples:
    ---------
    >>> data = ArrayData(x, y)
    >>> data_log = data.transform('log')
    >>> x_vals = data_log.get_x()
    """
    
    x: np.ndarray
    y: np.ndarray
    weights: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    _space: str = 'linear'
    
    def __post_init__(self):
        """Validate and prepare data after initialization."""
        # Convert to numpy arrays
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        
        # Validate shapes
        if self.x.shape != self.y.shape:
            raise ValueError(f"x and y must have same shape: {self.x.shape} vs {self.y.shape}")
        
        # Convert weights if provided
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != self.x.shape:
                raise ValueError(f"weights must have same shape as x: {self.weights.shape} vs {self.x.shape}")
        
        # Create default mask if not provided
        if self.mask is None:
            self.mask = np.isfinite(self.x) & np.isfinite(self.y) & (self.y > 0)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.x.shape:
                raise ValueError(f"mask must have same shape as x: {self.mask.shape} vs {self.x.shape}")
    
    def get_x(self) -> np.ndarray:
        """Get x values (masked)."""
        return self.x[self.mask]
    
    def get_y(self) -> np.ndarray:
        """Get y values (masked)."""
        return self.y[self.mask]
    
    def get_weights(self) -> Optional[np.ndarray]:
        """Get weights (masked) or None."""
        if self.weights is None:
            return None
        return self.weights[self.mask]
    
    def get_mask(self) -> np.ndarray:
        """Get validity mask."""
        return self.mask
    
    @property
    def space(self) -> str:
        """Current data space."""
        return self._space
    
    def transform(self, space: str) -> 'ArrayData':
        """
        Transform data to different space.
        
        Parameters:
        -----------
        space : str
            Target space: 'linear', 'log', 'log_log'
        
        Returns:
        --------
        ArrayData : new data object in transformed space
        """
        if space == self._space:
            return self
        
        x_new = self.x.copy()
        y_new = self.y.copy()
        
        # Apply transformation
        if space == 'log':
            # Only transform y
            y_new = np.log(np.maximum(y_new, 1e-12))
        
        elif space == 'log_log':
            # Transform both x and y
            x_new = np.log(np.maximum(x_new, 1e-12))
            y_new = np.log(np.maximum(y_new, 1e-12))
        
        elif space == 'linear':
            # Transform back from log
            if self._space == 'log':
                y_new = np.exp(y_new)
            elif self._space == 'log_log':
                x_new = np.exp(x_new)
                y_new = np.exp(y_new)
        
        else:
            raise ValueError(f"Unknown space: {space}")
        
        # Create new ArrayData object
        return ArrayData(
            x=x_new,
            y=y_new,
            weights=self.weights.copy() if self.weights is not None else None,
            mask=self.mask.copy(),
            _space=space
        )
    
    def set_mask(self, mask: np.ndarray):
        """
        Update the validity mask.
        
        Parameters:
        -----------
        mask : array-like
            New boolean mask
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.x.shape:
            raise ValueError(f"mask must have same shape as x: {mask.shape} vs {self.x.shape}")
        self.mask = mask
    
    def apply_range_mask(self, x_min: Optional[float] = None, 
                        x_max: Optional[float] = None):
        """
        Apply range-based mask (restrict to x_min <= x <= x_max).
        
        Parameters:
        -----------
        x_min : float, optional
            Minimum x value
        x_max : float, optional
            Maximum x value
        """
        new_mask = self.mask.copy()
        
        if x_min is not None:
            new_mask &= (self.x >= x_min)
        
        if x_max is not None:
            new_mask &= (self.x <= x_max)
        
        self.mask = new_mask
    
    def __len__(self) -> int:
        """Number of valid data points."""
        return np.sum(self.mask)
    
    def __repr__(self) -> str:
        n_valid = np.sum(self.mask)
        n_total = len(self.x)
        return f"ArrayData({n_valid}/{n_total} points, space='{self._space}')"