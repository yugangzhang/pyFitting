"""
pyFitting.core - Core Framework Components

This module contains the fundamental interfaces, types, and result containers
that form the foundation of the fitting framework.
"""

from pyFitting.core.interfaces import (
    IData,
    IModel,
    ILoss,
    IOptimizer,
    IEvaluator
)

from pyFitting.core.types import (
    FitSpace,
    OptimizerType,
    ParameterSet,
    OptimizeResult
)

from pyFitting.core.result import FitResult


__all__ = [
    # Interfaces
    'IData',
    'IModel',
    'ILoss',
    'IOptimizer',
    'IEvaluator',
    
    # Types
    'FitSpace',
    'OptimizerType',
    'ParameterSet',
    'OptimizeResult',
    
    # Result
    'FitResult'
]

__version__ = '0.1.0'