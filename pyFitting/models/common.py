"""
pyFitting.models.common - Common Models

This module provides commonly used models like Gaussian, Exponential, Polynomial, etc.
"""

import numpy as np
from typing import Dict

from pyFitting.models.base import BaseModel


__all__ = [
    'GaussianModel',
    'ExponentialModel',
    'LinearModel',
    'PowerLawModel',
    'PolynomialModel'
]


class GaussianModel(BaseModel):
    """
    Gaussian (normal) distribution model.
    
    Model: y = A * exp(-0.5 * ((x - mu) / sigma)^2) + c
    
    Parameters:
    -----------
    A : amplitude
    mu : center
    sigma : width
    c : offset
    """
    
    def evaluate(self, x: np.ndarray, A: float, mu: float, 
                sigma: float, c: float = 0.0) -> np.ndarray:
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + c
    
    def get_initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # Find peak
        idx_max = np.argmax(y)
        A = y[idx_max]
        mu = x[idx_max]
        
        # Estimate width from FWHM
        half_max = A / 2
        above_half = y > half_max
        if np.any(above_half):
            fwhm = np.ptp(x[above_half])
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = (x.max() - x.min()) / 10
        
        c = np.min(y)
        
        return {'A': A, 'mu': mu, 'sigma': sigma, 'c': c}


class ExponentialModel(BaseModel):
    """
    Exponential decay model.
    
    Model: y = A * exp(-k * x) + c
    
    Parameters:
    -----------
    A : amplitude
    k : decay rate
    c : offset
    """
    
    def evaluate(self, x: np.ndarray, A: float, k: float, c: float = 0.0) -> np.ndarray:
        return A * np.exp(-k * x) + c
    
    def get_initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        A = y.max() - y.min()
        c = y.min()
        
        # Estimate decay rate from half-life
        y_mid = (y.max() + y.min()) / 2
        idx = np.argmin(np.abs(y - y_mid))
        if idx > 0:
            k = np.log(2) / x[idx]
        else:
            k = 1.0 / (x.max() - x.min())
        
        return {'A': A, 'k': k, 'c': c}


class LinearModel(BaseModel):
    """
    Linear model.
    
    Model: y = a * x + b
    
    Parameters:
    -----------
    a : slope
    b : intercept
    """
    
    def evaluate(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b
    
    def get_initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # Simple linear regression
        a = np.cov(x, y)[0, 1] / np.var(x)
        b = np.mean(y) - a * np.mean(x)
        return {'a': a, 'b': b}


class PowerLawModel(BaseModel):
    """
    Power law model.
    
    Model: y = c3 * x^(-n)
    
    Useful for modeling backgrounds in scattering data (e.g., Porod law with n=4).
    
    Parameters:
    -----------
    c3 : amplitude
    n : exponent
    """
    
    def evaluate(self, x: np.ndarray, c3: float, n: float = 4.0) -> np.ndarray:
        return c3 * x**(-n)
    
    def get_initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # Fit in log-log space
        log_x = np.log(x + 1e-12)
        log_y = np.log(y + 1e-12)
        
        # Linear fit: log(y) = log(c3) - n * log(x)
        coeffs = np.polyfit(log_x, log_y, 1)
        n = -coeffs[0]
        log_c3 = coeffs[1]
        c3 = np.exp(log_c3)
        
        return {'c3': c3, 'n': n}


class PolynomialModel(BaseModel):
    """
    Polynomial model.
    
    Model: y = c0 + c1*x + c2*x^2 + ... + cn*x^n
    
    Parameters:
    -----------
    degree : int
        Polynomial degree
    **c0, c1, ..., cn : float
        Polynomial coefficients
    """
    
    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree
    
    def evaluate(self, x: np.ndarray, **params) -> np.ndarray:
        # Build coefficient array
        coeffs = np.array([params.get(f'c{i}', 0.0) for i in range(self.degree + 1)])
        return np.polyval(coeffs[::-1], x)
    
    def get_initial_guess(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        # Polynomial fit
        coeffs = np.polyfit(x, y, self.degree)
        # Reverse order (polyfit returns highest degree first)
        return {f'c{i}': float(coeffs[::-1][i]) for i in range(self.degree + 1)}




class DoubleGaussianModel(BaseModel):
    """
    Two Gaussian peaks.
    
    For complex models, data-driven initialization is ABSOLUTELY ESSENTIAL!
    """
    
    def evaluate(self, x, A1, mu1, sigma1, A2, mu2, sigma2, c):
        peak1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1)**2)
        peak2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2)**2)
        return peak1 + peak2 + c
    

    def get_initial_guess(self, x, y):
        """
        Intelligent data-driven guess for two peaks.
        
        Strategy:
        1. Find two local maxima in the data
        2. Estimate parameters for each peak
        3. Estimate baseline
        
        This is MUCH more robust than fixed defaults!
        """
        from scipy.signal import find_peaks
        
        # Find baseline
        c = np.percentile(y, 10)  # 10th percentile as baseline
        
        # Find peaks in data
        peaks, properties = find_peaks(y - c, prominence=0.1 * (np.max(y) - c))
        
        if len(peaks) >= 2:
            # Sort by prominence and take top 2
            prominences = properties['prominences']
            top_peaks = peaks[np.argsort(prominences)[-2:]]
            top_peaks = np.sort(top_peaks)  # Sort by position
            
            # Peak 1
            idx1 = top_peaks[0]
            A1 = y[idx1] - c
            mu1 = x[idx1]
            sigma1 = (x[-1] - x[0]) / 20  # Rough guess
            
            # Peak 2
            idx2 = top_peaks[1]
            A2 = y[idx2] - c
            mu2 = x[idx2]
            sigma2 = (x[-1] - x[0]) / 20
            
        elif len(peaks) == 1:
            # Only one peak found, place second peak elsewhere
            idx1 = peaks[0]
            A1 = y[idx1] - c
            mu1 = x[idx1]
            sigma1 = (x[-1] - x[0]) / 20
            
            # Place second peak at 1/3 of data range away
            A2 = A1 / 2
            mu2 = mu1 + (x[-1] - x[0]) / 3
            sigma2 = sigma1
            
        else:
            # No peaks found, use data-driven fallback
            A1 = (np.max(y) - c) / 2
            A2 = A1
            mu1 = x[len(x) // 3]
            mu2 = x[2 * len(x) // 3]
            sigma1 = (x[-1] - x[0]) / 10
            sigma2 = sigma1
        
        return {
            'A1': A1, 'mu1': mu1, 'sigma1': sigma1,
            'A2': A2, 'mu2': mu2, 'sigma2': sigma2,
            'c': c
        }


    