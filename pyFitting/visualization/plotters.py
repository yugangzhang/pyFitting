"""
pyFitting.visualization.plotters - General Visualization Functions

This module provides general-purpose plotting functions for fit results.
No matplotlib needed in main package - this is optional!
"""

import numpy as np
from typing import Optional, Tuple, List
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed. Visualization functions will not work.")


__all__ = [
    'plot_fit',
    'plot_residuals',
    'plot_fit_with_residuals',
    'plot_parameter_corners',
    'plot_diagnostics',
    'plot_comparison'
]




def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")



def plot_data(data, 
             figsize: Tuple[float, float] = (6,4),
             logx: bool = False,
             logy: bool = False,
             show_ci: bool = False,
             title: Optional[str] = None,
             xlabel: str = 'x',
             ylabel: str = 'y',
             save: Optional[str] = None,
             **kwargs):
    """
    Plot Data.
    
    Parameters:
    -----------
    Data:  ArrayData
    figsize : tuple
        Figure size (width, height)
    logx : bool
        Use log scale for x-axis
    logy : bool
        Use log scale for y-axis
    show_ci : bool
        Show confidence intervals (if available in result)
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    save : str, optional
        Filename to save plot
    **kwargs : dict
        Additional matplotlib kwargs
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    _check_matplotlib()
    
    # Get data
    x = data.get_x()
    y_data = data.get_y()
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.plot(x, y_data, 'o', markersize=4, alpha=0.6, label='Data', color='#1f77b4')
    
    # Set scales
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        title = f'Data'
    ax.set_title(title, fontsize=14)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5) 
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_fit(result, 
             figsize: Tuple[float, float] = (10, 6),
             logx: bool = False,
             logy: bool = False,
             show_ci: bool = False,
             title: Optional[str] = None,
             xlabel: str = 'x',
             ylabel: str = 'y',
             save: Optional[str] = None,
             **kwargs):
    """
    Plot fit results.
    
    Parameters:
    -----------
    result : FitResult
        Fit result to plot
    figsize : tuple
        Figure size (width, height)
    logx : bool
        Use log scale for x-axis
    logy : bool
        Use log scale for y-axis
    show_ci : bool
        Show confidence intervals (if available in result)
    title : str, optional
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    save : str, optional
        Filename to save plot
    **kwargs : dict
        Additional matplotlib kwargs
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    _check_matplotlib()
    
    # Get data
    x = result.data.get_x()
    y_data = result.data.get_y()
    y_fit = result.y_fit
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.plot(x, y_data, 'o', markersize=4, alpha=0.6, label='Data', color='#1f77b4')
    
    # Plot fit
    ax.plot(x, y_fit, '-', linewidth=2, label=f'Fit (R² = {result.metrics["r2"]:.4f})', 
            color='#ff7f0e')
    
    # Confidence intervals
    if show_ci and hasattr(result, 'confidence_bands') and result.confidence_bands is not None:
        ax.fill_between(x, result.confidence_bands['lower'], 
                        result.confidence_bands['upper'],
                        alpha=0.3, color='#ff7f0e', label='95% CI')
    
    # Set scales
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        title = f'Fit Results ({result.optimizer})'
    ax.set_title(title, fontsize=14)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Parameter text box
    param_text = "Parameters:\n"
    for name, value in list(result.parameters.values.items())[:5]:  # Show first 5
        error = result.parameter_errors.get(name, 0.0)
        param_text += f"{name} = {value:.4g} ± {error:.4g}\n"
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_residuals(result,
                   figsize: Tuple[float, float] = (10, 4),
                   relative: bool = True,
                   logx: bool = False,
                   xlabel: str = 'x',
                   title: Optional[str] = None,
                   save: Optional[str] = None,
                   **kwargs):
    """
    Plot residuals.
    
    Parameters:
    -----------
    result : FitResult
        Fit result
    figsize : tuple
        Figure size
    relative : bool
        Plot relative residuals (%) vs absolute
    logx : bool
        Use log scale for x-axis
    xlabel : str
        X-axis label
    title : str, optional
        Plot title
    save : str, optional
        Filename to save
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    _check_matplotlib()
    
    x = result.data.get_x()
    
    if relative:
        residuals = result.get_relative_residuals_percent()
        ylabel = 'Relative Residuals (%)'
        ylim = (-20, 20)
    else:
        residuals = result.get_residuals()
        ylabel = 'Residuals'
        ylim = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, residuals, 'o', markersize=4, alpha=0.6, color='#1f77b4')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    if logx:
        ax.set_xscale('log')
    
    if ylim:
        ax.set_ylim(ylim)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title is None:
        title = 'Residuals'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_fit_with_residuals(result,
                            figsize: Tuple[float, float] = (12, 8),
                            logx: bool = False,
                            logy: bool = False,
                            xlabel: str = 'x',
                            ylabel: str = 'y',
                            title: Optional[str] = None,
                            save: Optional[str] = None,
                            **kwargs):
    """
    Plot fit and residuals in a 2x1 layout.
    
    Parameters:
    -----------
    result : FitResult
        Fit result
    figsize : tuple
        Figure size
    logx : bool
        Use log scale for x-axis
    logy : bool
        Use log scale for y-axis (fit panel only)
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Overall title
    save : str, optional
        Filename to save
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    _check_matplotlib()
    
    # Get data
    x = result.data.get_x()
    y_data = result.data.get_y()
    y_fit = result.y_fit
    residuals_pct = result.get_relative_residuals_percent()
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[3, 1],
                          hspace=0.3, wspace=0.3)
    
    # Main fit plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y_data, 'o', markersize=4, alpha=0.6, label='Data')
    ax1.plot(x, y_fit, '-', linewidth=2, label=f'Fit (R² = {result.metrics["r2"]:.4f})')
    
    if logx:
        ax1.set_xscale('log')
    if logy:
        ax1.set_yscale('log')
    
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Parameter text
    param_text = "Parameters:\n"
    for name, value in result.parameters.values.items():
        error = result.parameter_errors.get(name, 0.0)
        param_text += f"{name} = {value:.4g} ± {error:.4g}\n"
    
    ax1.text(0.02, 0.02, param_text, transform=ax1.transAxes,
            verticalalignment='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Residuals plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, residuals_pct, 'o', markersize=4, alpha=0.6)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    
    if logx:
        ax2.set_xscale('log')
    
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('Residuals (%)', fontsize=12)
    ax2.set_ylim(-20, 20)
    ax2.grid(True, alpha=0.3)
    
    # Metrics panel
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    metrics_text = "Fit Quality:\n"
    metrics_text += f"R² = {result.metrics['r2']:.4f}\n"
    metrics_text += f"R² (log) = {result.metrics['r2_log']:.4f}\n"
    metrics_text += f"χ²/dof = {result.metrics['chi2_reduced']:.3f}\n"
    metrics_text += f"RMSE = {result.metrics['rmse']:.4g}\n"
    metrics_text += f"p-value = {result.metrics.get('p_value', 0):.3f}\n"
    
    ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
            verticalalignment='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    if title is None:
        title = f'Fit Analysis ({result.optimizer})'
    fig.suptitle(title, fontsize=14)
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig, [ax1, ax2, ax3]


def plot_parameter_corners(result,
                          figsize: Optional[Tuple[float, float]] = None,
                          n_samples: int = 1000,
                          title: Optional[str] = None,
                          save: Optional[str] = None,
                          **kwargs):
    """
    Plot parameter corner plot (pairwise distributions).
    
    Requires covariance matrix in result.
    
    Parameters:
    -----------
    result : FitResult
        Fit result with covariance matrix
    figsize : tuple, optional
        Figure size (auto-calculated if None)
    n_samples : int
        Number of samples to draw
    title : str, optional
        Plot title
    save : str, optional
        Filename to save
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    _check_matplotlib()
    
    if result.covariance is None:
        raise ValueError("Covariance matrix not available in result")
    
    param_names = list(result.parameters.values.keys())
    n_params = len(param_names)
    
    if figsize is None:
        figsize = (2 * n_params, 2 * n_params)
    
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)
    
    # Sample from parameter distribution
    param_values = np.array([result.parameters.values[name] for name in param_names])
    samples = np.random.multivariate_normal(param_values, result.covariance, n_samples)
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j] if n_params > 1 else axes
            
            if i == j:
                # Diagonal: histogram
                ax.hist(samples[:, i], bins=30, density=True, alpha=0.7, color='skyblue')
                ax.axvline(param_values[i], color='red', linestyle='--', linewidth=2)
                ax.set_ylabel('Density' if j == 0 else '')
                ax.set_yticks([])
            else:
                # Off-diagonal: scatter
                ax.scatter(samples[:, j], samples[:, i], alpha=0.3, s=5, color='steelblue')
                ax.plot(param_values[j], param_values[i], 'ro', markersize=8)
            
            # Labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=10)
            else:
                ax.set_xticks([])
            
            if j == 0 and i != j:
                ax.set_ylabel(param_names[i], fontsize=10)
            elif j != 0:
                ax.set_yticks([])
    
    if title is None:
        title = 'Parameter Correlations'
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_diagnostics(result,
                    figsize: Tuple[float, float] = (14, 10),
                    save: Optional[str] = None,
                    **kwargs):
    """
    Plot comprehensive diagnostics (4-panel).
    
    Panels:
    1. Fit
    2. Residuals
    3. Residual histogram
    4. Q-Q plot
    
    Parameters:
    -----------
    result : FitResult
        Fit result
    figsize : tuple
        Figure size
    save : str, optional
        Filename to save
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    _check_matplotlib()
    from scipy import stats
    
    x = result.data.get_x()
    y_data = result.data.get_y()
    y_fit = result.y_fit
    residuals = result.get_residuals()
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Fit
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y_data, 'o', markersize=4, alpha=0.6, label='Data')
    ax1.plot(x, y_fit, '-', linewidth=2, label='Fit')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'Fit (R² = {result.metrics["r2"]:.4f})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs x
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, residuals, 'o', markersize=4, alpha=0.6)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residuals vs x', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Residual histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    ax3.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2, label='Normal')
    
    ax3.set_xlabel('Residuals', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Residual Distribution', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(f'Diagnostic Plots ({result.optimizer})', fontsize=14, y=0.995)
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig, [ax1, ax2, ax3]


def plot_comparison(results: List,
                   labels: Optional[List[str]] = None,
                   figsize: Tuple[float, float] = (12, 6),
                   logx: bool = False,
                   logy: bool = False,
                   xlabel: str = 'x',
                   ylabel: str = 'y',
                   title: str = 'Model Comparison',
                   save: Optional[str] = None,
                   **kwargs):
    """
    Compare multiple fit results.
    
    Parameters:
    -----------
    results : list of FitResult
        List of fit results to compare
    labels : list of str, optional
        Labels for each result
    figsize : tuple
        Figure size
    logx : bool
        Use log scale for x-axis
    logy : bool
        Use log scale for y-axis
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    save : str, optional
        Filename to save
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    _check_matplotlib()
    
    if labels is None:
        labels = [f'Fit {i+1}' for i in range(len(results))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data from first result
    x = results[0].data.get_x()
    y_data = results[0].data.get_y()
    ax.plot(x, y_data, 'o', markersize=4, alpha=0.6, label='Data', color='gray')
    
    # Plot all fits
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (result, label) in enumerate(zip(results, labels)):
        r2 = result.metrics['r2']
        ax.plot(result.data.get_x(), result.y_fit, '-', linewidth=2,
               label=f'{label} (R²={r2:.4f})', color=colors[i])
    
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig, ax