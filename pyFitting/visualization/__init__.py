"""
pyFitting.visualization - Plotting Functions

This module provides visualization functions for fit results.

Note: matplotlib is an optional dependency. Install with:
    pip install matplotlib
"""

try:
    from pyFitting.visualization.plotters import (
        plot_data,
        plot_fit,
        plot_residuals,
        plot_fit_with_residuals,
        plot_parameter_corners,
        plot_diagnostics,
        plot_comparison
    )
    from pyFitting.visualization.plot import (
        plot1D,
        colors_, markers_, lstyles_,
        create_fig_ax, create_2ax_main_minor,
         
    )

    
    
    __all__ = [
        'plot_fit',
        'plot_residuals',
        'plot_fit_with_residuals',
        'plot_parameter_corners',
        'plot_diagnostics',
        'plot_comparison',
         'plot1D',


        
    ]
    
except ImportError:
    import warnings
    warnings.warn("matplotlib not installed. Visualization module will not work.")
    __all__ = []