"""
pyFitting.data.smoothing - Data Smoothing Methods

A comprehensive toolkit for smoothing noisy data with multiple algorithms
optimized for scientific data analysis, especially scattering data (SAXS/WAXS).

Author: Yugang Zhang
License: MIT
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from typing import Optional, Tuple
import warnings


__all__ = ['DataSmoother', 'quick_smooth', 'compare_smoothers']


class DataSmoother:
    """
    Comprehensive data smoothing with multiple methods.
    
    Supported Methods:
    ------------------
    - 'gaussian': Gaussian filter (fast, good for uniform noise)
    - 'savgol': Savitzky-Golay filter (preserves features, good for trends)
    - 'spline': Spline smoothing (flexible, excellent for SAXS)
    - 'moving_average': Simple moving average (fast, basic)
    - 'median': Median filter (excellent for outliers/spikes)
    
    Parameters:
    -----------
    method : str
        Smoothing method (default: 'savgol')
    
    Examples:
    ---------
    >>> smoother = DataSmoother('savgol')
    >>> y_smooth = smoother.smooth(x, y, window_length=11, polyorder=3)
    
    >>> # For SAXS data, smooth in log space
    >>> y_smooth = smoother.smooth_log_space(q, iq, window_length=9)
    """
    
    METHODS = ['gaussian', 'savgol', 'spline', 'moving_average', 'median']
    
    def __init__(self, method: str = 'savgol'):
        """Initialize with smoothing method."""
        if method not in self.METHODS:
            raise ValueError(f"Method must be one of {self.METHODS}")
        self.method = method
    
    def smooth(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Smooth data using selected method.
        
        Parameters:
        -----------
        x : array
            Independent variable
        y : array
            Dependent variable to smooth
        **kwargs : dict
            Method-specific parameters (see individual methods)
        
        Returns:
        --------
        y_smooth : array
            Smoothed data
        """
        methods = {
            'gaussian': self._gaussian_smooth,
            'savgol': self._savgol_smooth,
            'spline': self._spline_smooth,
            'moving_average': self._moving_average_smooth,
            'median': self._median_smooth
        }
        return methods[self.method](x, y, **kwargs)
    
    def _gaussian_smooth(self, x: np.ndarray, y: np.ndarray, 
                        sigma: float = 2.0) -> np.ndarray:
        """
        Gaussian filter smoothing.
        
        Parameters:
        -----------
        sigma : float
            Standard deviation for Gaussian kernel (1-5 typical)
            Higher = more smoothing
        
        Best for: Uniform Gaussian noise
        """
        return gaussian_filter1d(y, sigma=sigma, mode='nearest')
    
    def _savgol_smooth(self, x: np.ndarray, y: np.ndarray,
                      window_length: Optional[int] = None,
                      polyorder: int = 3) -> np.ndarray:
        """
        Savitzky-Golay filter (preserves peaks/dips better).
        
        Parameters:
        -----------
        window_length : int, optional
            Window size (must be odd). Auto-computed if None (~5% of data)
        polyorder : int
            Polynomial order (2-5 typical, must be < window_length)
        
        Best for: Preserving sharp features while removing noise
        """
        if window_length is None:
            window_length = max(5, int(len(y) * 0.05))
            if window_length % 2 == 0:
                window_length += 1
        
        window_length = max(polyorder + 2, window_length)
        if window_length % 2 == 0:
            window_length += 1
        
        window_length = min(window_length, len(y))
        if window_length < polyorder + 2:
            polyorder = max(1, window_length - 2)
        
        if window_length < 3:
            warnings.warn("Data too short for Savitzky-Golay filter, returning original")
            return y.copy()
        
        return savgol_filter(y, window_length, polyorder, mode='nearest')
    
    def _spline_smooth(self, x: np.ndarray, y: np.ndarray,
                      smoothing: Optional[float] = None,
                      spline_order: int = 3) -> np.ndarray:
        """
        Spline smoothing (best for SAXS data in log space).
        
        Parameters:
        -----------
        smoothing : float, optional
            Smoothing factor (s parameter). Auto-computed if None
            Higher = more smoothing, 0 = interpolation
        spline_order : int
            Spline degree (3-5 typical)
        
        Best for: Smooth underlying trends, SAXS/WAXS data
        """
        if smoothing is None:
            smoothing = len(x) * 0.5
        
        try:
            spline = UnivariateSpline(x, y, k=spline_order, s=smoothing)
            return spline(x)
        except Exception as e:
            warnings.warn(f"Spline smoothing failed: {e}, returning original")
            return y.copy()
    
    def _moving_average_smooth(self, x: np.ndarray, y: np.ndarray,
                               window: int = 5) -> np.ndarray:
        """
        Simple moving average.
        
        Parameters:
        -----------
        window : int
            Window size (3-10 typical)
        
        Best for: Quick smoothing, basic noise reduction
        """
        if window < 2:
            return y.copy()
        
        window = min(window, len(y))
        kernel = np.ones(window) / window
        y_padded = np.pad(y, (window//2, window//2), mode='edge')
        y_smooth = np.convolve(y_padded, kernel, mode='valid')
        
        return y_smooth[:len(y)]
    
    def _median_smooth(self, x: np.ndarray, y: np.ndarray,
                      window: int = 5) -> np.ndarray:
        """
        Median filter (robust to outliers).
        
        Parameters:
        -----------
        window : int
            Window size (3-7 typical, must be odd)
        
        Best for: Removing spikes, outliers, salt-and-pepper noise
        """
        if window % 2 == 0:
            window += 1
        
        window = min(window, len(y))
        if window < 3:
            return y.copy()
        
        return median_filter(y, size=window, mode='nearest')
    
    def smooth_log_space(self, x: np.ndarray, y: np.ndarray, 
                        **kwargs) -> np.ndarray:
        """
        Smooth in log-log space (recommended for SAXS/power-law data).
        
        Parameters:
        -----------
        x : array
            Independent variable
        y : array
            Dependent variable
        **kwargs : dict
            Method-specific parameters
        
        Returns:
        --------
        y_smooth : array
            Smoothed data (in linear space)
        
        Examples:
        ---------
        >>> smoother = DataSmoother('savgol')
        >>> iq_smooth = smoother.smooth_log_space(q, iq, window_length=11)
        """
        log_x = np.log(np.maximum(x, 1e-10))
        log_y = np.log(np.maximum(y, 1e-10))
        
        log_y_smooth = self.smooth(log_x, log_y, **kwargs)
        
        return np.exp(log_y_smooth)
    
    def __repr__(self) -> str:
        return f"DataSmoother(method='{self.method}')"


def quick_smooth(x: np.ndarray, y: np.ndarray, 
                method: str = 'savgol',
                log_space: bool = False,
                **kwargs) -> np.ndarray:
    """
    Quick smoothing function for convenience.
    
    Parameters:
    -----------
    x, y : arrays
        Data to smooth
    method : str
        Smoothing method: 'gaussian', 'savgol', 'spline', etc.
    log_space : bool
        If True, smooth in log-log space (for SAXS/power-law data)
    **kwargs : dict
        Method-specific parameters
    
    Returns:
    --------
    y_smooth : array
        Smoothed data
    
    Examples:
    ---------
    >>> # Gaussian smoothing
    >>> y_smooth = quick_smooth(x, y, method='gaussian', sigma=2.0)
    
    >>> # Savitzky-Golay with custom window
    >>> y_smooth = quick_smooth(x, y, method='savgol', 
    ...                         window_length=11, polyorder=3)
    
    >>> # SAXS data (smooth in log space)
    >>> iq_smooth = quick_smooth(q, iq, method='savgol', log_space=True)
    """
    smoother = DataSmoother(method=method)
    
    if log_space:
        return smoother.smooth_log_space(x, y, **kwargs)
    else:
        return smoother.smooth(x, y, **kwargs)


def compare_smoothers(x: np.ndarray, y: np.ndarray,
                     methods: Optional[list] = None,
                     log_space: bool = False,
                     **kwargs) -> dict:
    """
    Compare multiple smoothing methods on the same data.
    
    Parameters:
    -----------
    x, y : arrays
        Data to smooth
    methods : list, optional
        List of methods to compare (default: all methods)
    log_space : bool
        If True, smooth in log space
    **kwargs : dict
        Common parameters for all methods
    
    Returns:
    --------
    results : dict
        Dictionary of {method: smoothed_data}
    
    Examples:
    ---------
    >>> results = compare_smoothers(x, y_noisy, 
    ...                            methods=['gaussian', 'savgol', 'median'])
    >>> for method, y_smooth in results.items():
    ...     plt.plot(x, y_smooth, label=method)
    """
    if methods is None:
        methods = DataSmoother.METHODS
    
    results = {}
    for method in methods:
        try:
            results[method] = quick_smooth(x, y, method=method, 
                                          log_space=log_space, **kwargs)
        except Exception as e:
            warnings.warn(f"Method {method} failed: {e}")
    
    return results


######################################
def visualize_weights(q, iq, weights):
    """
    Visualize the computed weights overlaid on data.
    Helps you understand what's being emphasized.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top: Data with weight overlay
    ax1.loglog(q, iq, 'b-', linewidth=2, label='I(q)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(q, weights * len(weights), 'r--', linewidth=2, 
                  label='Weights (normalized)', alpha=0.7)
    ax1_twin.fill_between(q, 0, weights * len(weights), alpha=0.2, color='red')
    
    ax1.set_ylabel('I(q)', fontsize=12, color='b')
    ax1_twin.set_ylabel('Weight (relative)', fontsize=12, color='r')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('Data and Feature-Based Weights', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Weighted residual emphasis
    ax2.plot(q, weights * len(weights), 'r-', linewidth=2)
    ax2.fill_between(q, 0, weights * len(weights), alpha=0.3, color='red')
    ax2.set_xlabel('q (Å⁻¹)', fontsize=12)
    ax2.set_ylabel('Relative Weight', fontsize=12)
    ax2.set_title('Weight Distribution (high = more emphasis)', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


'''
Example:
#iq_masked_smooth = quick_smooth( q_masked, iq_masked,  method  = 'savgol',  window_length=11, polyorder=3 , log_space= True )  # 
iq_masked_smooth = quick_smooth( q_masked, iq_masked,  method  = 'gaussian',  sigma = 3 , log_space= True )  #   
#iq_masked_smooth = quick_smooth( q_masked, iq_masked,  method  = 'spline',  spline_order = 6 , log_space= True )  #  NOT GOOD: 
#iq_masked_smooth = quick_smooth( q_masked, iq_masked,  method  = 'moving_average',  window = 11 , log_space= True ) 
#iq_masked_smooth = quick_smooth( q_masked, iq_masked,  method  = 'median',  window = 11 , log_space= True ) 


fig = plt.figure(figsize=[6,4])
ax = fig.add_subplot(111)
plot1D( x= q_masked, y = iq_masked,  ax=ax, m='d', ls='', c='k', logy=F, logx=T, markersize=3, legend='data')
plot1D( x= q_masked, y = iq_masked_smooth,  ax=ax, m='', ls='-', c='r', logy=T, logx=T, markersize=3, legend='smooth')



'''

    