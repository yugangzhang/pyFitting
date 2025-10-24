"""
pyFitting.models - Mathematical Models

This module provides model classes for fitting.
"""

from pyFitting.models.base import BaseModel
from pyFitting.models.common import (
    GaussianModel,
    ExponentialModel,
    LinearModel,
    PowerLawModel,
    PolynomialModel
)


__all__ = [
    'BaseModel',
    'GaussianModel',
    'ExponentialModel',
    'LinearModel',
    'PowerLawModel',
    'PolynomialModel'
]