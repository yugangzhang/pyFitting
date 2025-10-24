"""
pyFitting.loss - Loss Functions

This module provides loss functions for fitting.
"""

from pyFitting.loss.standard import (
    MSELoss,
    Chi2Loss,
    CorrelationLoss,
    HybridLoss
)


__all__ = [
    'MSELoss',
    'Chi2Loss',
    'CorrelationLoss',
    'HybridLoss'
]
 