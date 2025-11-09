"""
Comprehensive examples for DataSmoother with various noise types.

This script demonstrates all smoothing methods on simulated data with
different types of noise commonly encountered in scientific data.
"""

import numpy as np
import matplotlib.pyplot as plt
from smoothing import DataSmoother, quick_smooth, compare_smoothers


def generate_test_signal(n_points=200, signal_type='saxs'):
    """
    Generate test signals for demonstration.
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    signal_type : str
        'saxs' - SAXS-like scattering curve
        'peaks' - Multiple peaks
        'sine' - Sinusoidal
        'step' - Step function
    
    Returns:
    --------
    x, y_clean : arrays
        Clean signal without noise
    """
    if signal_type == 'saxs':
        # Simulate SAXS curve: form factor + Porod tail
        q = np.logspace(-2, 0, n_points)
        R = 100  # radius in Angstroms
        sigma = 0.1  # polydispersity
        
        # Sphere form factor with polydispersity
        def sphere_ff(q, R):
            qR = q * R
            ff = 3 * (np.sin(qR) - qR * np.cos(qR)) / qR**3
            return ff**2
        
        # Schulz distribution
        z = 1 / sigma**2 - 1
        R_dist = np.linspace(R * 0.5, R * 1.5, 50)
        weights = (z + 1)**(z + 1) / np.math.factorial(z) * \
                 (R_dist / R)**z * np.exp(-(z + 1) * R_dist / R)
        weights /= weights.sum()
        
        # Average form factor
        I = np.zeros_like(q)
        for Ri, wi in zip(R_dist, weights):
            I += wi * sphere_ff(q, Ri)
        
        # Add Porod tail
        I += 0.01 * q**(-4)
        
        # Add scale
        I *= 1e6
        
        return q, I
    
    elif signal_type == 'peaks':
        # Multiple Gaussian peaks
        x = np.linspace(0, 10, n_points)
        y = (np.exp(-((x - 2) / 0.5)**2) + 
             0.7 * np.exp(-((x - 5) / 0.3)**2) +
             0.5 * np.exp(-((x - 7.5) / 0.6)**2))
        return x, y
    
    elif signal_type == 'sine':
        # Sinusoidal with trend
        x = np.linspace(0, 4 * np.pi, n_points)
        y = np.sin(x) + 0.5 * np.sin(3 * x) + 0.1 * x
        return x, y
    
    elif signal_type == 'step':
        # Step function with smooth transitions
        x = np.linspace(0, 10, n_points)
        y = np.zeros_like(x)
        y[x > 2] = 1.0
        y[x > 5] = 0.5
        y[x > 8] = 1.5
        # Smooth the steps slightly
        from scipy.ndimage import gaussian_filter1d
        y = gaussian_filter1d(y, sigma=2)
        return x, y
    
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def add_noise(y_clean, noise_type='gaussian', noise_level=0.1, **kwargs):
    """
    Add various types of noise to clean signal.
    
    Parameters:
    -----------
    y_clean : array
        Clean signal
    noise_type : str
        'gaussian' - Gaussian/normal noise
        'poisson' - Poisson noise (photon counting)
        'outliers' - Random outliers/spikes
        'salt_pepper' - Salt and pepper noise
        'mixed' - Combination of multiple noise types
    noise_level : float
        Noise amplitude (relative to signal)
    
    Returns:
    --------
    y_noisy : array
        Signal with added noise
    """
    n = len(y_clean)
    
    if noise_type == 'gaussian':
        # Gaussian white noise
        noise = noise_level * np.std(y_clean) * np.random.randn(n)
        return y_clean + noise
    
    elif noise_type == 'poisson':
        # Poisson noise (photon counting statistics)
        # Scale to have reasonable counts
        scale = kwargs.get('scale', 1000)
        y_scaled = y_clean * scale
        y_noisy = np.random.poisson(y_scaled) / scale
        return y_noisy
    
    elif noise_type == 'outliers':
        # Random outliers (spikes)
        y_noisy = y_clean.copy()
        n_outliers = kwargs.get('n_outliers', int(0.05 * n))
        outlier_indices = np.random.choice(n, n_outliers, replace=False)
        outlier_magnitude = kwargs.get('outlier_magnitude', 5.0)
        y_noisy[outlier_indices] += outlier_magnitude * noise_level * np.std(y_clean) * \
                                     np.random.randn(n_outliers)
        return y_noisy
    
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        y_noisy = y_clean.copy()
        p = kwargs.get('p', 0.05)  # probability of corruption
        
        # Salt (bright spots)
        salt_mask = np.random.rand(n) < p / 2
        y_noisy[salt_mask] = np.max(y_clean) * (1 + noise_level)
        
        # Pepper (dark spots)
        pepper_mask = np.random.rand(n) < p / 2
        y_noisy[pepper_mask] = np.min(y_clean) * (1 - noise_level)
        
        return y_noisy
    
    elif noise_type == 'mixed':
        # Combination of noise types
        y_noisy = y_clean.copy()
        
        # Add Gaussian noise
        y_noisy += 0.5 * noise_level * np.std(y_clean) * np.random.randn(n)
        
        # Add some outliers
        n_outliers = int(0.02 * n)
        outlier_indices = np.random.choice(n, n_outliers, replace=False)
        y_noisy[outlier_indices] += 3 * noise_level * np.std(y_clean) * \
                                    np.random.randn(n_outliers)
        
        return y_noisy
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def example_1_basic_smoothing():
    """Example 1: Basic smoothing with different methods."""
    print("=" * 70)
    print("Example 1: Comparing Different Smoothing Methods")
    print("=" * 70)
    
    # Generate noisy data
    x, y_clean = generate_test_signal(n_points=200, signal_type='peaks')
    y_noisy = add_noise(y_clean, noise_type='gaussian', noise_level=0.15)
    
    # Compare all smoothing methods
    results = compare_smoothers(x, y_noisy)
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (method, y_smooth) in enumerate(results.items()):
        ax = axes[idx]
        ax.plot(x, y_clean, 'g-', linewidth=2, label='Clean signal', alpha=0.7)
        ax.plot(x, y_noisy, 'k.', markersize=3, label='Noisy data', alpha=0.3)
        ax.plot(x, y_smooth, 'r-', linewidth=2, label=f'{method} smoothed')
        
        # Calculate error
        rmse = np.sqrt(np.mean((y_smooth - y_clean)**2))
        ax.set_title(f'{method.capitalize()} (RMSE={rmse:.4f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplot
    if len(results) < 6:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('example1_basic_smoothing.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Example 1 complete: Saved as 'example1_basic_smoothing.png'\n")


def example_2_noise_types():
    """Example 2: Different noise types and appropriate smoothers."""
    print("=" * 70)
    print("Example 2: Different Noise Types")
    print("=" * 70)
    
    # Generate clean signal
    x, y_clean = generate_test_signal(n_points=200, signal_type='sine')
    
    noise_types = ['gaussian', 'poisson', 'outliers', 'salt_pepper']
    best_methods = ['savgol', 'gaussian', 'median', 'median']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (noise_type, best_method) in enumerate(zip(noise_types, best_methods)):
        # Add noise
        if noise_type == 'poisson':
            y_noisy = add_noise(y_clean, noise_type=noise_type, scale=500)
        else:
            y_noisy = add_noise(y_clean, noise_type=noise_type, noise_level=0.2)
        
        # Smooth with best method
        if best_method == 'savgol':
            y_smooth = quick_smooth(x, y_noisy, method=best_method, 
                                   window_length=15, polyorder=3)
        elif best_method == 'median':
            y_smooth = quick_smooth(x, y_noisy, method=best_method, window=5)
        else:
            y_smooth = quick_smooth(x, y_noisy, method=best_method, sigma=2.5)
        
        # Plot
        ax = axes[idx]
        ax.plot(x, y_clean, 'g-', linewidth=2.5, label='Clean signal', alpha=0.8)
        ax.plot(x, y_noisy, 'k.', markersize=4, label=f'{noise_type} noise', alpha=0.4)
        ax.plot(x, y_smooth, 'r-', linewidth=2, label=f'{best_method} smoothed')
        
        rmse = np.sqrt(np.mean((y_smooth - y_clean)**2))
        ax.set_title(f'{noise_type.capitalize()} Noise → {best_method.capitalize()} Filter\n(RMSE={rmse:.4f})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('example2_noise_types.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Example 2 complete: Saved as 'example2_noise_types.png'\n")


def example_3_saxs_data():
    """Example 3: SAXS data smoothing in log space."""
    print("=" * 70)
    print("Example 3: SAXS Data Smoothing (Log Space)")
    print("=" * 70)
    
    # Generate SAXS-like data
    q, I_clean = generate_test_signal(n_points=150, signal_type='saxs')
    
    # Add Poisson noise (typical for photon counting)
    I_noisy = add_noise(I_clean, noise_type='poisson', scale=10000)
    
    # Add some outliers (cosmic rays, etc.)
    I_noisy = add_noise(I_noisy, noise_type='outliers', noise_level=0.3, 
                       n_outliers=10, outlier_magnitude=10)
    
    # Smooth in linear space vs log space
    smoother = DataSmoother('savgol')
    
    # Linear space
    I_smooth_linear = smoother.smooth(q, I_noisy, window_length=11, polyorder=3)
    
    # Log space (recommended for SAXS)
    I_smooth_log = smoother.smooth_log_space(q, I_noisy, window_length=11, polyorder=3)
    
    # Also try median filter for outlier removal
    I_smooth_median = quick_smooth(q, I_noisy, method='median', 
                                  log_space=True, window=5)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Log-log plot
    ax1.loglog(q, I_clean, 'g-', linewidth=2.5, label='Clean signal', alpha=0.8)
    ax1.loglog(q, I_noisy, 'k.', markersize=4, label='Noisy data', alpha=0.3)
    ax1.loglog(q, I_smooth_linear, 'b--', linewidth=2, label='Smoothed (linear space)', alpha=0.7)
    ax1.loglog(q, I_smooth_log, 'r-', linewidth=2, label='Smoothed (log space)')
    
    ax1.set_xlabel('q (Å⁻¹)', fontsize=12)
    ax1.set_ylabel('I(q)', fontsize=12)
    ax1.set_title('SAXS Data: Linear vs Log Space Smoothing', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2.semilogx(q, I_clean - I_smooth_linear, 'b-', linewidth=2, 
                label='Residuals (linear space)', alpha=0.7)
    ax2.semilogx(q, I_clean - I_smooth_log, 'r-', linewidth=2, 
                label='Residuals (log space)')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel('q (Å⁻¹)', fontsize=12)
    ax2.set_ylabel('I_clean - I_smooth', fontsize=12)
    ax2.set_title('Smoothing Residuals', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example3_saxs_smoothing.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Example 3 complete: Saved as 'example3_saxs_smoothing.png'\n")
    print("Key insight: For SAXS data, always smooth in LOG SPACE!")
    print("  - Preserves power-law behavior at high-q")
    print("  - Better handles data spanning many orders of magnitude\n")


def example_4_parameter_tuning():
    """Example 4: Effect of parameter tuning."""
    print("=" * 70)
    print("Example 4: Parameter Tuning Effects")
    print("=" * 70)
    
    # Generate noisy data
    x, y_clean = generate_test_signal(n_points=200, signal_type='peaks')
    y_noisy = add_noise(y_clean, noise_type='gaussian', noise_level=0.2)
    
    # Test different parameters for Savitzky-Golay
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Window length effect
    ax = axes[0, 0]
    for window in [5, 11, 21, 31]:
        y_smooth = quick_smooth(x, y_noisy, method='savgol', 
                               window_length=window, polyorder=3)
        ax.plot(x, y_smooth, linewidth=2, label=f'window={window}')
    ax.plot(x, y_clean, 'k--', linewidth=1.5, label='Clean', alpha=0.5)
    ax.set_title('Savitzky-Golay: Window Length Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Polynomial order effect
    ax = axes[0, 1]
    for poly in [1, 2, 3, 5]:
        y_smooth = quick_smooth(x, y_noisy, method='savgol', 
                               window_length=11, polyorder=poly)
        ax.plot(x, y_smooth, linewidth=2, label=f'polyorder={poly}')
    ax.plot(x, y_clean, 'k--', linewidth=1.5, label='Clean', alpha=0.5)
    ax.set_title('Savitzky-Golay: Polynomial Order Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gaussian sigma effect
    ax = axes[1, 0]
    for sigma in [0.5, 1.0, 2.0, 4.0]:
        y_smooth = quick_smooth(x, y_noisy, method='gaussian', sigma=sigma)
        ax.plot(x, y_smooth, linewidth=2, label=f'sigma={sigma}')
    ax.plot(x, y_clean, 'k--', linewidth=1.5, label='Clean', alpha=0.5)
    ax.set_title('Gaussian: Sigma Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spline smoothing factor effect
    ax = axes[1, 1]
    for s_factor in [0.1, 0.5, 1.0, 2.0]:
        y_smooth = quick_smooth(x, y_noisy, method='spline', 
                               smoothing=len(x) * s_factor)
        ax.plot(x, y_smooth, linewidth=2, label=f's_factor={s_factor}')
    ax.plot(x, y_clean, 'k--', linewidth=1.5, label='Clean', alpha=0.5)
    ax.set_title('Spline: Smoothing Factor Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example4_parameter_tuning.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Example 4 complete: Saved as 'example4_parameter_tuning.png'\n")
    print("Parameter selection guidelines:")
    print("  - Larger window/sigma → more smoothing, may lose features")
    print("  - Smaller window/sigma → less smoothing, may keep noise")
    print("  - Start with defaults, adjust based on your noise level\n")


def example_5_step_function():
    """Example 5: Preserving sharp features (edges)."""
    print("=" * 70)
    print("Example 5: Preserving Sharp Features")
    print("=" * 70)
    
    # Generate step function with noise
    x, y_clean = generate_test_signal(n_points=200, signal_type='step')
    y_noisy = add_noise(y_clean, noise_type='gaussian', noise_level=0.15)
    
    # Compare methods on step function
    methods_to_test = {
        'Gaussian': {'method': 'gaussian', 'sigma': 3.0},
        'Savitzky-Golay': {'method': 'savgol', 'window_length': 15, 'polyorder': 2},
        'Median': {'method': 'median', 'window': 5},
        'Spline': {'method': 'spline', 'smoothing': len(x) * 0.3}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (name, params) in enumerate(methods_to_test.items()):
        y_smooth = quick_smooth(x, y_noisy, **params)
        
        ax = axes[idx]
        ax.plot(x, y_clean, 'g-', linewidth=3, label='Clean signal', alpha=0.7)
        ax.plot(x, y_noisy, 'k.', markersize=4, label='Noisy data', alpha=0.3)
        ax.plot(x, y_smooth, 'r-', linewidth=2, label=f'{name} smoothed')
        
        ax.set_title(f'{name} Filter on Step Function')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('example5_sharp_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Example 5 complete: Saved as 'example5_sharp_features.png'\n")
    print("Best methods for preserving edges:")
    print("  - Median filter: Best for sharp edges")
    print("  - Savitzky-Golay: Good compromise")
    print("  - Avoid Gaussian/Spline: They blur edges\n")


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("RUNNING ALL SMOOTHING EXAMPLES")
    print("=" * 70 + "\n")
    
    np.random.seed(42)  # For reproducibility
    
    example_1_basic_smoothing()
    example_2_noise_types()
    example_3_saxs_data()
    example_4_parameter_tuning()
    example_5_step_function()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - example1_basic_smoothing.png")
    print("  - example2_noise_types.png")
    print("  - example3_saxs_smoothing.png")
    print("  - example4_parameter_tuning.png")
    print("  - example5_sharp_features.png")
    print("\nCheck these images to understand which smoother works best for your data!\n")

'''
if __name__ == "__main__":
    run_all_examples()
'''\

    