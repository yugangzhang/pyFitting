"""
pyFitting.utils.common - Common Utility Functions

This module provides utility functions for data analysis and processing.
"""

import numpy as np
from typing import Tuple


__all__ = [
    'get_ab_correlation',
    'get_similarity_by_overlap'
]


def get_ab_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """
    Get correlation of two 1D arrays.
    
    Derived from pandas.DataFrame.corrwith method.
    
    Formula:
        A = sum((a - a.mean()) * (b - b.mean()))
        B = (len(a) - 1) * a.std() * b.std()
        Cor = A / B
    
    Parameters:
    -----------
    a : array-like
        First 1D array
    b : array-like
        Second 1D array
    
    Returns:
    --------
    float : correlation coefficient
    
    Examples:
    ---------
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([2, 4, 6, 8, 10])
    >>> corr = get_ab_correlation(a, b)
    >>> print(f"Correlation: {corr:.4f}")
    Correlation: 1.0000
    
    Notes:
    ------
    Originally written by YG 2018/11/05
    """
    a = np.array(a)
    b = np.array(b)
    
    if len(a) != len(b):
        raise ValueError(f"Arrays must have same length: {len(a)} vs {len(b)}")
    
    if len(a) < 2:
        raise ValueError("Arrays must have at least 2 elements")
    
    # Compute correlation
    numerator = ((a - a.mean()) * (b - b.mean())).sum()
    denominator = (len(a) - 1) * a.std(ddof=1) * b.std(ddof=1)
    
    if denominator < 1e-30:
        return 0.0
    
    return numerator / denominator


def get_similarity_by_overlap(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Compute similarity by overlap coefficient.
    
    Overlap coefficient = I12 / (I11 + I22 - |I12|)
    
    where:
        I12 = dot(y1, y2)
        I11 = dot(y1, y1)
        I22 = dot(y2, y2)
    
    This measures the similarity between two vectors.
    Value ranges from 0 (no overlap) to 1 (perfect overlap).
    
    Parameters:
    -----------
    y1 : array-like
        First array
    y2 : array-like
        Second array
    
    Returns:
    --------
    float : overlap coefficient (0 to 1)
    
    Examples:
    ---------
    >>> y1 = np.array([1, 2, 3, 4])
    >>> y2 = np.array([1, 2, 3, 4])
    >>> overlap = get_similarity_by_overlap(y1, y2)
    >>> print(f"Overlap: {overlap:.4f}")
    Overlap: 1.0000
    
    >>> y1 = np.array([1, 0, 0, 0])
    >>> y2 = np.array([0, 1, 0, 0])
    >>> overlap = get_similarity_by_overlap(y1, y2)
    >>> print(f"Overlap: {overlap:.4f}")
    Overlap: 0.0000
    """
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    if y1.shape != y2.shape:
        raise ValueError(f"Arrays must have same shape: {y1.shape} vs {y2.shape}")
    
    I12 = np.dot(y1, y2)
    I11 = np.dot(y1, y1)
    I22 = np.dot(y2, y2)
    
    denominator = I11 + I22 - abs(I12)
    
    if denominator < 1e-30:
        return 0.0
    
    return I12 / denominator