"""
pyFitting.models - Mathematical Models

This module provides model classes for fitting.
"""

from pyFitting.model.base import BaseModel
from pyFitting.model.common import (
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