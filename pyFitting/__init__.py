"""
pyFitting - Modular Fitting Framework

A clean, modular, extensible framework for curve fitting.

Quick Start:
------------
>>> from pyFitting import Fitter, ArrayData, GaussianModel
>>> 
>>> # Your data
>>> data = ArrayData(x, y)
>>> 
>>> # Choose model
>>> model = GaussianModel()
>>> 
>>> # Fit
>>> fitter = Fitter(data, model)
>>> result = fitter.fit()
>>> 
>>> # View results
>>> result.summary()

Components:
-----------
- Data:       ArrayData, ...
- Models:     GaussianModel, ExponentialModel, LinearModel, ...
- Loss:       MSELoss, Chi2Loss, CorrelationLoss, ...
- Optimizers: LocalOptimizer, ...
- Fitter:     Main fitting interface

For more examples, see:
    examples/simple_fit.py
    examples/saxs_fit.py
"""

# Core
from pyFitting.core import (
    FitSpace,
    OptimizerType,
    ParameterSet,
    OptimizeResult,
    FitResult
)

# Data
from pyFitting.data import ArrayData

# Models
from pyFitting.models import (
    BaseModel,
    GaussianModel,
    ExponentialModel,
    LinearModel,
    PowerLawModel,
    PolynomialModel
)

# Loss
from pyFitting.loss import (
    MSELoss,
    Chi2Loss,
    CorrelationLoss,
    HybridLoss
)

# Optimizers
from pyFitting.optimizers import (
    LocalOptimizer,
    compare_optimizers
)

# Evaluation
from pyFitting.evaluation import StandardEvaluator

# Workflow
from pyFitting.workflow import Fitter


__version__ = '0.1.0'
__author__ = 'pyFitting Team'
__license__ = 'MIT'


__all__ = [
    # Main interface
    'Fitter',
    
    # Data
    'ArrayData',
    
    # Models
    'BaseModel',
    'GaussianModel',
    'ExponentialModel',
    'LinearModel',
    'PowerLawModel',
    'PolynomialModel',
    
    # Loss
    'MSELoss',
    'Chi2Loss',
    'CorrelationLoss',
    'HybridLoss',
    
    # Optimizers
    'LocalOptimizer',
    'compare_optimizers',
    
    # Evaluation
    'StandardEvaluator',
    
    # Core types
    'FitSpace',
    'OptimizerType',
    'ParameterSet',
    'OptimizeResult',
    'FitResult',
]