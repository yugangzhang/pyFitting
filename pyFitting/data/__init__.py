"""
pyFitting.data - Data Handling

This module provides data containers and transformation utilities.
"""

from pyFitting.data.array_data import ArrayData
from pyFitting.data.smoothing import  DataSmoother, quick_smooth



__all__ = [
    'ArrayData', 'DataSmoother'
]