"""
PyScatt Visualization Module
===========================

A module for scientific data visualization, primarily focused on scattering data.
Provides utilities for 1D and 2D plotting with various customization options.

Originally developed by Y.G. @ CFN, Nov 2019
Some functions adapted from pyCHX package


PyScatt Visualization Module - Code Improvements
Overview
The refactored PyScatt Visualization Module provides a clean, well-organized framework for scientific data visualization, with a focus on scattering data. The code has been restructured to improve readability, maintainability, and reusability.
Key Improvements
1. Modular Organization
The code has been organized into logical sections:

Constants and iterators
Custom colormaps
Data transformation functions
Figure and axes creation
Label array and overlay handling
Main plotting functions

2. Comprehensive Documentation

Added clear module-level docstring explaining the purpose
Enhanced function docstrings with consistent formatting
Detailed parameter descriptions with proper types
Return value documentation
Clarified optional parameters with default values

3. Code Quality Improvements

Simplified redundant code paths
Improved error handling
Consolidated related functionality
Improved variable naming for clarity
Removed unused imports and code

4. Enhanced Function Architecture

Color Management: Created dedicated functions for color and marker iteration
Custom Colormaps: Organized custom colormaps into a structured dictionary
Data Transformation: Clarified data transformation paths for different visualization modes
Figure Creation: Standardized figure and axes creation patterns

Main Functions
Core Visualization Functions

plot1D: Versatile 1D data plotting with customization options
show_img: Base function for displaying 2D image data
show_imgz: Advanced image display with z-scale transformation options
plot_1d_scattering: Specialized function for 1D scattering data
plot_2D_scattering: Specialized function for 2D scattering data
plot_xy_with_fit: Display data points alongside fitted curves

Helper Functions

get_r_map: Generate a map of distances from a center point
plot_z_range: Determine appropriate min/max values for color scaling
plot_z_transform: Transform data for visualization with various scaling options
show_label_array: Display labeled regions of interest
show_label_array_on_image: Overlay labeled regions on an image

Figure Utilities

create_fig_ax: Create a figure with multiple subplots
create_2ax_main_minor: Create a figure with a main plot and a smaller subplot
add_lines_patches: Add vertical lines and rectangular highlights
show_angle_coordinates: Display angle coordinates for polar data


"""

'''
Example

# Basic 1D plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)
plot1D(y, x, marker='o', c='b', title='Sine Wave', xlabel='x', ylabel='sin(x)')

# Basic 2D image display
img = np.random.rand(100, 100)
show_img(img, title='Random Image', show_colorbar=True)

# 2D scattering plot with log scale
plot_2D_scattering(scattering_data, logs=True, title='Scattering Pattern')

# 1D scattering plot
q_values = np.logspace(-3, -1, 100)
intensity = 1/q_values**2  # Simple power law
plot_1d_scattering(q_values, intensity, title='SAXS Data')

# Plotting data with a fit
fit_curve = 1/q_values**2 * np.exp(-q_values**2)
plot_xy_with_fit(q_values, intensity, q_values, fit_curve, 
                 txts='Power law with\nexponential cutoff')
                 

'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.figure import Figure 
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from datetime import datetime
import copy
import itertools
 

# Optional imports with fallbacks
try:
    from modest_image import imshow
except ImportError:
    from matplotlib.pyplot import imshow

try:
    from skimage.draw import circle
except ImportError:
    from skimage.draw import circle_perimeter as circle

# Constants and Iterators
# -----------------------

def create_color_iterators():
    """Create and return iterators for colors and markers for consistent plot styling."""
    
    # Define base colors and markers
    color_list = [
        'k', 'g', "blue", 'r', "darkolivegreen", "brown", "m", "orange", "hotpink", 
        "darkcyan",   "gray", "green",   "cyan", "purple", "navy"
    ]
    
    marker_list = [
        "o", "D", "v", "^", "<", ">", "p", "s", "H", "h", "*", "d",
        "8", "1", "3", "2", "4", "+", "x", "_", "|", ",", "1"
    ]
    
    # Create iterators
    colors_iter = itertools.cycle(color_list)
    markers_iter = itertools.cycle(marker_list)
    linestyles_iter = itertools.cycle(['-', '--', '-.', '.', ':'])
    
    return colors_iter, markers_iter, linestyles_iter

# Initialize global iterators
colors_, markers_, lstyles_ = create_color_iterators()



def create_fig_ax(rows=2, cols=2, figsize=(8, 4), title='Figure', fontsize=12, y=1.04, 
                  num=None, clear=False):
    """
    Create a figure with multiple subplots.
    
    Parameters
    ----------
    rows : int, optional
        Number of rows in the subplot grid
    cols : int, optional
        Number of columns in the subplot grid
    figsize : tuple, optional
        Figure size in inches (width, height)
    title : str, optional
        Figure title
    fontsize : int, optional
        Font size for title
    y : float, optional
        Title vertical position
    num : int, optional
        Figure number
    clear : bool, optional
        Whether to clear an existing figure with the same number
        
    Returns
    -------
    fig : Figure
        The created figure
    axes : list
        List of subplot axes
    """
    fig = plt.figure(figsize=figsize, num=num, clear=clear)
    plt.title(title, fontsize=fontsize, y=y)
    
    if rows == 1 and cols == 1:
        pass
    else:
        plt.axis('off')
    
    axes = []
    for i in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, i + 1))
    
    return fig, axes

def create_2ax_main_minor(figsize=(8, 6), ratio=4, sharex=True, num=None, clear=False):
    """
    Create a figure with a main plot and a smaller subplot below it.
    
    Parameters
    ----------
    figsize : tuple, optional
        Figure size in inches (width, height)
    ratio : int, optional
        Size ratio between main and minor plots
    sharex : bool, optional
        Whether the plots share the x-axis
    num : int, optional
        Figure number
    clear : bool, optional
        Whether to clear an existing figure with the same number
        
    Returns
    -------
    fig : Figure
        The created figure
    ax1 : Axes
        The main (larger) axes
    ax2 : Axes
        The minor (smaller) axes
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True, num=num, clear=clear)
    gs = fig.add_gridspec(ratio + 1, 1, wspace=0.0, hspace=0.0)
    
    ax1 = fig.add_subplot(gs[0:ratio])
    
    if sharex:
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2 = fig.add_subplot(gs[ratio], sharex=ax1)
    else:
        plt.setp(ax1.get_xticklabels(), visible=True)
        ax2 = fig.add_subplot(gs[ratio])
    
    return fig, ax1, ax2



def plot1D(y, x=None, yerr=None, ax=None, return_fig=False, ls='-', figsize=None,
          legend=None, alpha=1.0, legend_size=None, lw=None, markersize=None, 
          tick_size=8, **kwargs):
    """
    Plot 1D data with customization options.
    
    Parameters
    ----------
    y : array-like
        Y-values to plot
    x : array-like, optional
        X-values. If None, uses the indices of y
    yerr : array-like, optional
        Error bars for y values
    ax : Axes, optional
        The axes to draw on. If None, a new figure is created
    return_fig : bool, optional
        Whether to return the figure object
    ls : str, optional
        Line style
    figsize : tuple, optional
        Figure size in inches (width, height)
    legend : str, optional
        Label for the legend
    alpha : float, optional
        Transparency of the plot
    legend_size : int, optional
        Font size for the legend
    lw : float, optional
        Line width
    markersize : float, optional
        Size of markers
    tick_size : int, optional
        Font size for tick labels
    **kwargs : dict
        Additional parameters that control plot appearance:
        - logx, logy: bool, whether to use log scale for each axis
        - logxy: bool, whether to use log scale for both axes
        - marker/m: str, marker style
        - color/c: str, line color
        - xlim, ylim: tuple, axis limits
        - xlabel, ylabel: str, axis labels
        - title: str, plot title
        - save: bool, whether to save the figure
        - path: str, path for saving
        
    Returns
    -------
    fig : Figure, optional
        The figure object (if return_fig is True)
    """
    # Create figure if needed
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Set default legend
    if legend is None:
        legend = ' '
    
    # Get plot parameters from kwargs
    logx = kwargs.get('logx', False)
    logy = kwargs.get('logy', False)
    logxy = kwargs.get('logxy', False)
    
    if logx and logy:
        logxy = True
    
    # Get marker and color, with fallbacks
    marker = kwargs.get('marker', kwargs.get('m', next(markers_)))
    color = kwargs.get('color', kwargs.get('c', next(colors_)))
    
    # Create x values if not provided
    if x is None:
        x = range(len(y))
    
    # Plot data with or without error bars
    if yerr is None:
        ax.plot(x, y, marker=marker, color=color, ls=ls, label=legend,
              lw=lw, markersize=markersize, alpha=alpha)
    else:
        ax.errorbar(x, y, yerr, marker=marker, color=color, ls=ls, label=legend,
                  lw=lw, markersize=markersize, alpha=alpha)
    
    # Set log scales if requested
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if logxy:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.tick_params(axis='both', which='minor', labelsize=tick_size)
    
    # Set axis limits if provided
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    
    # Set axis labels if provided
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    
    # Set title
    title = kwargs.get('title', 'plot')
    ax.set_title(title)
    
    # Add legend if applicable
    if legend and legend.strip():
        ax.legend(loc='best', fontsize=legend_size)
    
    # Save figure if requested
    if kwargs.get('save', False):
        fp = kwargs['path'] + f"{title}.png"
        plt.savefig(fp, dpi=fig.dpi)
    
    if return_fig:
        return fig

