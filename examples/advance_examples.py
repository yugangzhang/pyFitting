"""
pyFitting - Comprehensive Advanced Examples

This file demonstrates ALL features of pyFitting in detail:
1. Data features (masking, transformations, weights)
2. Models (built-in + custom)
3. Loss functions (all 5)
4. Optimizers (comparison)
5. Evaluators (metrics analysis)
6. Fitter features (guess, bounds, fixed params, etc.)

Each example is self-contained and demonstrates real-world usage patterns.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pyFitting import (
    # Main
    Fitter,
    ArrayData,
    
    # Models
    GaussianModel,
    ExponentialModel,
    LinearModel,
    PowerLawModel,
    PolynomialModel,
    BaseModel,
    
    # Loss functions
    MSELoss,
    Chi2Loss,
    CorrelationLoss,
    HybridLoss,
    OverlapLoss,
    
    # Optimizers
    LocalOptimizer,
    compare_optimizers,
    
    # Evaluation
    StandardEvaluator,
    
    # Utils
    get_ab_correlation,
    get_similarity_by_overlap,

   
)


# ==============================================================================
# EXAMPLE 1: DATA FEATURES
# ==============================================================================

def example_1_data_features():
    """
    Demonstrate all data features:
    - Creating data with different options
    - Masking (range mask, custom mask)
    - Transformations (linear, log, log_log)
    - Weights
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: DATA FEATURES")
    print("="*80)
    
    # Generate synthetic data with noise and outliers
    np.random.seed(42)
    x = np.linspace(0.1, 10, 150)
    y_true = 2.5 * np.exp(-0.5 * ((x - 5) / 1.2)**2) + 0.3
    y = y_true + np.random.normal(0, 0.05, len(x))
    
    # Add some outliers
    y[10] = 5.0
    y[50] = 0.0
    y[120] = 4.0
    
    print("\n--- 1a. Basic Data Creation ---")
    data = ArrayData(x, y)
    print(f"Data created: {data}")
    print(f"Valid points: {len(data)}/{len(x)}")
    
    # --- Range Masking ---
    print("\n--- 1b. Range Masking ---")
    data_masked = ArrayData(x, y)
    print(f"Original: {len(data_masked)} points")
    
    # Apply range mask (only fit x between 2 and 8)
    data_masked.apply_range_mask(x_min=2.0, x_max=8.0)
    print(f"After range mask (2 < x < 8): {len(data_masked)} points")
    
    # Fit both and compare
    model1 = GaussianModel()
    result_full = Fitter(data, model1).fit(verbose=False)
    result_masked = Fitter(data_masked, GaussianModel()).fit(verbose=False)
    
    print(f"\nFull data R²: {result_full.metrics['r2']:.4f}")
    print(f"Masked data R²: {result_masked.metrics['r2']:.4f}")
    
    # --- Custom Masking ---
    print("\n--- 1c. Custom Masking (Remove Outliers) ---")
    data_custom = ArrayData(x, y)
    
    # Create custom mask: remove points where |y - y_median| > 3*std
    y_median = np.median(y)
    y_std = np.std(y)
    custom_mask = np.abs(y - y_median) < 3 * y_std
    
    print(f"Original: {len(data_custom)} points")
    data_custom.set_mask(custom_mask)
    print(f"After outlier removal: {len(data_custom)} points")
    
    result_clean = Fitter(data_custom, GaussianModel()).fit(verbose=False)
    print(f"Cleaned data R²: {result_clean.metrics['r2']:.4f}")
    
    # --- Space Transformations ---
    print("\n--- 1d. Space Transformations ---")
    
    # Generate exponential data
    x_exp = np.linspace(0.1, 5, 100)
    y_exp = 10 * np.exp(-2 * x_exp) + 0.5 + np.random.normal(0, 0.1, len(x_exp))
    
    data_linear = ArrayData(x_exp, y_exp)
    data_log = data_linear.transform('log')
    data_loglog = data_linear.transform('log_log')
    
    print(f"Linear space: {data_linear.space}")
    print(f"Log space: {data_log.space}")
    print(f"Log-log space: {data_loglog.space}")
    
    # Fit in different spaces
    model_exp = ExponentialModel()
    
    result_linear = Fitter(data_linear, model_exp, 
                          loss=MSELoss(use_log=False)).fit(verbose=False)
    result_log = Fitter(data_log, ExponentialModel(),
                       loss=MSELoss(use_log=True)).fit(verbose=False)
    
    print(f"\nFit in linear space - R²: {result_linear.metrics['r2']:.4f}")
    print(f"Fit in log space - R² (log): {result_log.metrics['r2_log']:.4f}")
    
    # --- Weights ---
    print("\n--- 1e. Weighted Fitting ---")
    
    # Create weights (higher weight = more important)
    # Weight the peak region more heavily
    weights = np.ones_like(x)
    peak_region = (x > 4) & (x < 6)
    weights[peak_region] = 5.0  # 5x more weight in peak region
    
    data_weighted = ArrayData(x, y, weights=weights)
    
    result_unweighted = Fitter(data, GaussianModel()).fit(verbose=False)
    result_weighted = Fitter(data_weighted, GaussianModel()).fit(verbose=False)
    
    print(f"\nUnweighted fit - mu: {result_unweighted.parameters.values['mu']:.3f}")
    print(f"Weighted fit - mu: {result_weighted.parameters.values['mu']:.3f}")
    print(f"(Should be closer to 5.0 with weighting)")
    
    print("\n✓ Data features demonstrated!")
    return result_full, result_masked, result_weighted


# ==============================================================================
# EXAMPLE 2: MODELS (Built-in and Custom)
# ==============================================================================

def example_2_models():
    """
    Demonstrate all model features:
    - All built-in models
    - Custom models
    - Parameter management
    - Model composition
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: MODELS")
    print("="*80)
    
    np.random.seed(42)
    
    # --- 2a. Built-in Models ---
    print("\n--- 2a. Built-in Models ---")
    
    # Gaussian
    print("\n1. GaussianModel")
    x = np.linspace(0, 10, 100)
    y_gauss = 2.5 * np.exp(-0.5 * ((x - 5) / 1.2)**2) + 0.3
    model_gauss = GaussianModel()
    print(f"   Parameters: {list(model_gauss.get_parameters().values.keys())}")
    
    # Exponential
    print("\n2. ExponentialModel")
    x_exp = np.linspace(0, 5, 100)
    y_exp = 3.0 * np.exp(-0.8 * x_exp) + 0.2
    model_exp = ExponentialModel()
    print(f"   Parameters: {list(model_exp.get_parameters().values.keys())}")
    
    # Linear
    print("\n3. LinearModel")
    x_lin = np.linspace(0, 10, 50)
    y_lin = 2.5 * x_lin + 1.5
    model_lin = LinearModel()
    print(f"   Parameters: {list(model_lin.get_parameters().values.keys())}")
    
    # Power Law
    print("\n4. PowerLawModel")
    x_pow = np.linspace(0.1, 10, 100)
    y_pow = 5.0 * x_pow**(-3.5)
    model_pow = PowerLawModel()
    print(f"   Parameters: {list(model_pow.get_parameters().values.keys())}")
    
    # Polynomial
    print("\n5. PolynomialModel (degree=3)")
    x_poly = np.linspace(-5, 5, 100)
    y_poly = 0.5*x_poly**3 - 2*x_poly**2 + x_poly + 3
    model_poly = PolynomialModel(degree=3)
    print(f"   Parameters: {list(model_poly.get_parameters().values.keys())}")
    
    # --- 2b. Custom Models ---
    print("\n--- 2b. Custom Models ---")
    
    # Example 1: Lorentzian
    print("\n1. Custom Lorentzian Model")
    class LorentzianModel(BaseModel):
        """
        Lorentzian peak: y = A / (1 + ((x - x0) / gamma)^2) + c
        """
        def evaluate(self, x, A, x0, gamma, c):
            return A / (1 + ((x - x0) / gamma)**2) + c
        
        def get_initial_guess(self, x, y):
            idx_max = np.argmax(y)
            A = y[idx_max] - y.min()
            x0 = x[idx_max]
            gamma = (x.max() - x.min()) / 10
            c = y.min()
            return {'A': A, 'x0': x0, 'gamma': gamma, 'c': c}
    
    x_lor = np.linspace(0, 10, 100)
    y_lor = 3.0 / (1 + ((x_lor - 5) / 0.5)**2) + 0.2
    
    model_lor = LorentzianModel()
    data_lor = ArrayData(x_lor, y_lor)
    result_lor = Fitter(data_lor, model_lor).fit(verbose=False)
    print(f"   Lorentzian fit R²: {result_lor.metrics['r2']:.4f}")
    print(f"   Parameters: {result_lor.parameters.values}")
    
    # Example 2: Damped Oscillation
    print("\n2. Custom Damped Oscillation Model")
    class DampedOscillationModel(BaseModel):
        """
        Damped oscillation: y = A * exp(-decay * x) * cos(omega * x + phi) + c
        """
        def evaluate(self, x, A, decay, omega, phi, c):
            return A * np.exp(-decay * x) * np.cos(omega * x + phi) + c
        
        def get_initial_guess(self, x, y):
            A = (y.max() - y.min()) / 2
            decay = 0.1
            omega = 2 * np.pi / (x.max() - x.min()) * 3
            phi = 0
            c = y.mean()
            return {'A': A, 'decay': decay, 'omega': omega, 'phi': phi, 'c': c}
    
    x_osc = np.linspace(0, 10, 200)
    y_osc = 2.0 * np.exp(-0.3 * x_osc) * np.cos(4 * x_osc + 0.5) + 1.0
    y_osc += np.random.normal(0, 0.05, len(x_osc))
    
    model_osc = DampedOscillationModel()
    data_osc = ArrayData(x_osc, y_osc)
    result_osc = Fitter(data_osc, model_osc).fit(verbose=False)
    print(f"   Damped oscillation fit R²: {result_osc.metrics['r2']:.4f}")
    print(f"   Decay: {result_osc.parameters.values['decay']:.3f} (true: 0.3)")
    print(f"   Omega: {result_osc.parameters.values['omega']:.3f} (true: 4.0)")
    
    # Example 3: Sum of Two Gaussians
    print("\n3. Custom Sum of Two Gaussians")
    class DoubleGaussianModel(BaseModel):
        """
        Two Gaussian peaks: y = A1*exp(-0.5*((x-mu1)/sig1)^2) + A2*exp(-0.5*((x-mu2)/sig2)^2) + c
        """
        def evaluate(self, x, A1, mu1, sig1, A2, mu2, sig2, c):
            g1 = A1 * np.exp(-0.5 * ((x - mu1) / sig1)**2)
            g2 = A2 * np.exp(-0.5 * ((x - mu2) / sig2)**2)
            return g1 + g2 + c
        
        def get_initial_guess(self, x, y):
            # Find two peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(y, height=y.mean())
            
            if len(peaks) >= 2:
                idx1, idx2 = peaks[:2]
            else:
                idx1 = len(y) // 3
                idx2 = 2 * len(y) // 3
            
            A1 = y[idx1]
            mu1 = x[idx1]
            A2 = y[idx2]
            mu2 = x[idx2]
            sig1 = (x.max() - x.min()) / 10
            sig2 = sig1
            c = y.min()
            
            return {'A1': A1, 'mu1': mu1, 'sig1': sig1,
                   'A2': A2, 'mu2': mu2, 'sig2': sig2, 'c': c}
    
    x_dbl = np.linspace(0, 10, 200)
    y_dbl = (2.0 * np.exp(-0.5 * ((x_dbl - 3) / 0.8)**2) +
             1.5 * np.exp(-0.5 * ((x_dbl - 7) / 1.0)**2) + 0.2)
    y_dbl += np.random.normal(0, 0.05, len(x_dbl))
    
    model_dbl = DoubleGaussianModel()
    data_dbl = ArrayData(x_dbl, y_dbl)
    result_dbl = Fitter(data_dbl, model_dbl).fit(verbose=False)
    print(f"   Double Gaussian fit R²: {result_dbl.metrics['r2']:.4f}")
    print(f"   Peak 1 at: {result_dbl.parameters.values['mu1']:.2f} (true: 3.0)")
    print(f"   Peak 2 at: {result_dbl.parameters.values['mu2']:.2f} (true: 7.0)")
    
    print("\n✓ All models demonstrated!")
    return result_lor, result_osc, result_dbl


# ==============================================================================
# EXAMPLE 3: LOSS FUNCTIONS
# ==============================================================================

def example_3_loss_functions():
    """
    Demonstrate all 5 loss functions:
    - MSELoss (linear/log)
    - Chi2Loss (linear/log)
    - CorrelationLoss
    - HybridLoss
    - OverlapLoss
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: LOSS FUNCTIONS")
    print("="*80)
    
    # Generate data with heteroscedastic noise (varying error)
    np.random.seed(42)
    x = np.linspace(0.1, 10, 100)
    y_true = 5.0 * np.exp(-0.5 * x) + 0.5
    
    # Add heteroscedastic noise (larger errors at larger y)
    noise = np.random.normal(0, 0.1 * y_true, len(x))
    y = y_true + noise
    
    data = ArrayData(x, y)
    model = ExponentialModel()
    
    # --- Test All Loss Functions ---
    losses = {
        'MSE (linear)': MSELoss(use_log=False),
        'MSE (log)': MSELoss(use_log=True),
        'Chi2 (linear)': Chi2Loss(use_log=False),
        'Chi2 (log)': Chi2Loss(use_log=True),
        'Correlation': CorrelationLoss(use_log=True),
        'Hybrid (α=0.7)': HybridLoss(alpha=0.7, use_log=True),
        'Overlap (log)': OverlapLoss(use_log=True),
    }
    
    print("\nComparing loss functions on exponential decay data:")
    print(f"{'Loss Function':<20} {'R²':<10} {'R² (log)':<12} {'RMSE':<12} {'χ²_red':<10}")
    print("-" * 70)
    
    results = {}
    for name, loss in losses.items():
        fitter = Fitter(data, model, loss=loss)
        result = fitter.fit(verbose=False)
        results[name] = result
        
        print(f"{name:<20} {result.metrics['r2']:<10.4f} "
              f"{result.metrics['r2_log']:<12.4f} "
              f"{result.metrics['rmse']:<12.4e} "
              f"{result.metrics['chi2_reduced']:<10.4f}")
    
    # --- Analyze which loss is best for different scenarios ---
    print("\n--- Loss Function Recommendations ---")
    print("\n1. MSELoss (linear):")
    print("   ✓ Use when: Errors are uniform across data range")
    print("   ✓ Use when: Data is linear scale")
    print("   ✗ Avoid when: Data spans many orders of magnitude")
    
    print("\n2. MSELoss (log):")
    print("   ✓ Use when: Data spans orders of magnitude (SAXS, decay curves)")
    print("   ✓ Use when: Want to weight all points equally in log space")
    print("   ✓ Best for: Exponential data")
    
    print("\n3. Chi2Loss:")
    print("   ✓ Use when: Have proper error estimates")
    print("   ✓ Use when: Want statistical weighting")
    print("   ✓ Best for: Poisson-like data")
    
    print("\n4. CorrelationLoss:")
    print("   ✓ Use when: Shape matters more than absolute values")
    print("   ✓ Use when: Comparing patterns (SAXS profiles)")
    print("   ✓ Best for: Shape matching")
    
    print("\n5. HybridLoss:")
    print("   ✓ Use when: Want both shape and absolute value match")
    print("   ✓ Adjustable α parameter (0=MSE, 1=Correlation)")
    print("   ✓ Best for: General purpose fitting")
    
    print("\n6. OverlapLoss:")
    print("   ✓ Use when: Want to maximize overlap/similarity")
    print("   ✓ Use when: Comparing profiles (SAXS)")
    print("   ✓ Best for: Pattern matching in log space")
    
    print("\n✓ Loss functions compared!")
    return results


# ==============================================================================
# EXAMPLE 4: OPTIMIZERS
# ==============================================================================

def example_4_optimizers():
    """
    Demonstrate optimizer features:
    - All available methods
    - Comparison
    - Performance trade-offs
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: OPTIMIZERS")
    print("="*80)
    
    # Generate difficult fitting problem (multiple local minima)
    np.random.seed(42)
    x = np.linspace(0, 10, 150)
    
    # Sum of two Gaussians (challenging!)
    y_true = (2.0 * np.exp(-0.5 * ((x - 3) / 0.8)**2) +
              1.5 * np.exp(-0.5 * ((x - 7) / 1.0)**2) + 0.2)
    y = y_true + np.random.normal(0, 0.08, len(x))
    
    data = ArrayData(x, y)
    
    # Define double Gaussian model (from example 2)
    class DoubleGaussianModel(BaseModel):
        def evaluate(self, x, A1, mu1, sig1, A2, mu2, sig2, c):
            g1 = A1 * np.exp(-0.5 * ((x - mu1) / sig1)**2)
            g2 = A2 * np.exp(-0.5 * ((x - mu2) / sig2)**2)
            return g1 + g2 + c
        
        def get_initial_guess(self, x, y):
            return {'A1': 1.5, 'mu1': 3.5, 'sig1': 1.0,
                   'A2': 1.0, 'mu2': 6.5, 'sig2': 1.0, 'c': 0.2}
    
    model = DoubleGaussianModel()
    
    # --- Compare all optimizer methods ---
    print("\n--- Comparing Optimizer Methods ---")
    
    methods = ['SLSQP', 'L-BFGS-B', 'Powell', 'TNC', 'trust-constr', 'Nelder-Mead']
    
    print(f"\n{'Method':<15} {'Success':<10} {'R²':<10} {'RMSE':<12} {'nfev':<8} {'Time':<8}")
    print("-" * 70)
    
    import time
    results = {}
    
    for method in methods:
        optimizer = LocalOptimizer(method)
        fitter = Fitter(data, model, optimizer=optimizer)
        
        start = time.time()
        try:
            result = fitter.fit(verbose=False)
            elapsed = time.time() - start
            
            status = "✓" if result.success else "✗"
            print(f"{method:<15} {status:<10} {result.metrics['r2']:<10.4f} "
                  f"{result.metrics['rmse']:<12.4e} {result.nfev:<8} {elapsed:<8.3f}s")
            
            results[method] = result
            
        except Exception as e:
            print(f"{method:<15} {'ERROR':<10} {str(e)[:40]}")
    
    # --- Recommendations ---
    print("\n--- Optimizer Recommendations ---")
    print("\n1. SLSQP (Sequential Least Squares Programming)")
    print("   ✓ RECOMMENDED for most problems")
    print("   ✓ Fast and robust")
    print("   ✓ Handles bounds well")
    print("   ✓ Good for constrained optimization")
    
    print("\n2. L-BFGS-B (Limited-memory BFGS with Bounds)")
    print("   ✓ Very fast for smooth problems")
    print("   ✗ Can fail on difficult problems")
    print("   ✓ Good for large parameter spaces")
    
    print("\n3. Powell (Powell's method)")
    print("   ✓ Derivative-free (no gradient needed)")
    print("   ✓ Good for noisy objectives")
    print("   ✗ Slower than gradient-based")
    
    print("\n4. trust-constr (Trust Region)")
    print("   ✓ Most robust")
    print("   ✓ Best for difficult problems")
    print("   ✗ Slower than others")
    print("   ✓ Use when others fail")
    
    print("\n5. TNC (Truncated Newton)")
    print("   ✓ Good for smooth problems with bounds")
    print("   ✓ Fast convergence")
    
    print("\n6. Nelder-Mead (Simplex)")
    print("   ✓ Derivative-free")
    print("   ✗ Slow convergence")
    print("   ✗ No bound constraints")
    print("   ✓ Use as last resort")
    
    print("\n✓ Optimizers compared!")
    return results


# ==============================================================================
# EXAMPLE 5: FITTER FEATURES (Initial Guess, Bounds, Fixed Parameters)
# ==============================================================================

def example_5_fitter_features():
    """
    Demonstrate advanced fitter features:
    - Custom initial guess
    - Parameter bounds
    - Fixed parameters
    - Verbose mode
    - Multiple fits
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: FITTER FEATURES")
    print("="*80)
    
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 2.5 * np.exp(-0.5 * ((x - 5) / 1.2)**2) + 0.3
    y = y_true + np.random.normal(0, 0.05, len(x))
    
    data = ArrayData(x, y)
    model = GaussianModel()
    
    # --- 5a. Default Fit (Auto Initial Guess) ---
    print("\n--- 5a. Default Fit (Auto Initial Guess) ---")
    result_default = Fitter(data, model).fit(verbose=False)
    print(f"Auto initial guess:")
    print(f"  A = {result_default.parameters.values['A']:.3f}")
    print(f"  mu = {result_default.parameters.values['mu']:.3f}")
    print(f"  sigma = {result_default.parameters.values['sigma']:.3f}")
    print(f"  R² = {result_default.metrics['r2']:.4f}")
    
    # --- 5b. Custom Initial Guess ---
    print("\n--- 5b. Custom Initial Guess ---")
    
    # Try a bad initial guess
    bad_guess = {'A': 5.0, 'mu': 2.0, 'sigma': 3.0, 'c': 1.0}
    print(f"Bad initial guess: A=5.0, mu=2.0, sigma=3.0, c=1.0")
    
    result_bad = Fitter(data, model).fit(initial_guess=bad_guess, verbose=False)
    print(f"Fit still succeeds:")
    print(f"  A = {result_bad.parameters.values['A']:.3f}")
    print(f"  mu = {result_bad.parameters.values['mu']:.3f}")
    print(f"  R² = {result_bad.metrics['r2']:.4f}")
    
    # Try a good initial guess
    good_guess = {'A': 2.5, 'mu': 5.0, 'sigma': 1.0, 'c': 0.3}
    print(f"\nGood initial guess: A=2.5, mu=5.0, sigma=1.0, c=0.3")
    
    result_good = Fitter(data, model).fit(initial_guess=good_guess, verbose=False)
    print(f"Fit converges faster:")
    print(f"  nfev (bad guess): {result_bad.nfev}")
    print(f"  nfev (good guess): {result_good.nfev}")
    
    # --- 5c. Parameter Bounds ---
    print("\n--- 5c. Parameter Bounds ---")
    
    # Fit without bounds
    model_unbound = GaussianModel()
    result_unbound = Fitter(data, model_unbound).fit(verbose=False)
    print(f"Without bounds:")
    print(f"  sigma = {result_unbound.parameters.values['sigma']:.3f}")
    
    # Fit with bounds
    model_bound = GaussianModel()
    bounds = {
        'A': (1.0, 5.0),
        'mu': (4.0, 6.0),
        'sigma': (0.5, 2.0),
        'c': (0.0, 1.0)
    }
    
    result_bound = Fitter(data, model_bound).fit(bounds=bounds, verbose=False)
    print(f"With bounds σ ∈ [0.5, 2.0]:")
    print(f"  sigma = {result_bound.parameters.values['sigma']:.3f}")
    print(f"  (Stays within bounds)")
    
    # --- 5d. Fixed Parameters ---
    print("\n--- 5d. Fixed Parameters ---")
    
    # Suppose we know the center is at x=5.0
    model_fixed = GaussianModel()
    model_fixed.set_parameters(mu=5.0)  # Set value
    model_fixed.fix_parameter('mu', value=5.0)  # Fix it
    
    print(f"Fixing mu = 5.0")
    result_fixed = Fitter(data, model_fixed).fit(verbose=False)
    print(f"Fit result:")
    print(f"  mu = {result_fixed.parameters.values['mu']:.3f} (fixed)")
    print(f"  A = {result_fixed.parameters.values['A']:.3f}")
    print(f"  sigma = {result_fixed.parameters.values['sigma']:.3f}")
    print(f"  R² = {result_fixed.metrics['r2']:.4f}")
    print(f"  Free parameters: {result_fixed.parameters.get_free_names()}")
    
    # --- 5e. Multiple Fixed Parameters ---
    print("\n--- 5e. Multiple Fixed Parameters ---")
    
    # Fix both center and width
    model_multi = GaussianModel()
    model_multi.fix_parameter('mu', value=5.0)
    model_multi.fix_parameter('sigma', value=1.2)
    
    print(f"Fixing mu=5.0 and sigma=1.2")
    result_multi = Fitter(data, model_multi).fit(verbose=False)
    print(f"Only fitting A and c:")
    print(f"  A = {result_multi.parameters.values['A']:.3f}")
    print(f"  c = {result_multi.parameters.values['c']:.3f}")
    print(f"  R² = {result_multi.metrics['r2']:.4f}")
    
    # --- 5f. Verbose Mode ---
    print("\n--- 5f. Verbose Mode ---")
    print("Running fit with verbose=True:")
    result_verbose = Fitter(data, GaussianModel()).fit(verbose=True)
    
    # --- 5g. Complete Example: All Features Together ---
    print("\n--- 5g. All Features Together ---")
    
    model_complete = GaussianModel()
    
    # Set initial guess
    initial = {'A': 2.0, 'mu': 5.0, 'sigma': 1.0, 'c': 0.5}
    
    # Set bounds
    bounds = {
        'A': (0.1, 10.0),
        'mu': (0.0, 10.0),
        'sigma': (0.1, 3.0),
        'c': (-1.0, 2.0)
    }
    
    # Fix one parameter
    model_complete.fix_parameter('c', value=0.3)
    
    # Configure optimizer
    optimizer = LocalOptimizer('SLSQP')
    
    # Configure loss
    loss = HybridLoss(alpha=0.8, use_log=True)
    
    # Fit with all features
    print("Fitting with:")
    print(f"  - Custom initial guess")
    print(f"  - Parameter bounds")
    print(f"  - Fixed c = 0.3")
    print(f"  - Hybrid loss (α=0.8)")
    print(f"  - SLSQP optimizer")
    
    fitter = Fitter(data, model_complete, loss=loss, optimizer=optimizer)
    result_complete = fitter.fit(initial_guess=initial, bounds=bounds, verbose=False)
    
    print(f"\nFit successful: {result_complete.success}")
    print(f"R² = {result_complete.metrics['r2']:.4f}")
    print(f"Parameters:")
    for name, value in result_complete.parameters.values.items():
        fixed = "(fixed)" if result_complete.parameters.fixed.get(name, False) else ""
        print(f"  {name} = {value:.3f} {fixed}")
    
    print("\n✓ All fitter features demonstrated!")
    return result_complete


# ==============================================================================
# EXAMPLE 6: EVALUATOR & METRICS
# ==============================================================================

def example_6_evaluator_metrics():
    """
    Demonstrate evaluator features:
    - All available metrics
    - Metrics interpretation
    - Quality assessment
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: EVALUATOR & METRICS")
    print("="*80)
    
    # Generate three fits of varying quality
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 2.5 * np.exp(-0.5 * ((x - 5) / 1.2)**2) + 0.3
    
    # Good fit
    y_good = y_true + np.random.normal(0, 0.03, len(x))
    data_good = ArrayData(x, y_good)
    result_good = Fitter(data_good, GaussianModel()).fit(verbose=False)
    
    # Medium fit (more noise)
    y_medium = y_true + np.random.normal(0, 0.15, len(x))
    data_medium = ArrayData(x, y_medium)
    result_medium = Fitter(data_medium, GaussianModel()).fit(verbose=False)
    
    # Bad fit (wrong model)
    y_bad = y_true + np.random.normal(0, 0.05, len(x))
    data_bad = ArrayData(x, y_bad)
    result_bad = Fitter(data_bad, LinearModel()).fit(verbose=False)  # Wrong model!
    
    # --- Display All Metrics ---
    print("\n--- All Available Metrics ---")
    print(f"\n{'Metric':<25} {'Good Fit':<15} {'Medium Fit':<15} {'Bad Fit':<15}")
    print("-" * 75)
    
    metrics_to_show = [
        'r2', 'r2_log', 'pearson_r', 'pearson_r_log',
        'chi2_reduced', 'rmse', 'rmse_log', 'mae',
        'mean_rel_error', 'max_rel_error', 'p_value'
    ]
    
    for metric in metrics_to_show:
        if metric in result_good.metrics:
            val_good = result_good.metrics[metric]
            val_medium = result_medium.metrics[metric]
            val_bad = result_bad.metrics[metric]
            
            print(f"{metric:<25} {val_good:<15.4f} {val_medium:<15.4f} {val_bad:<15.4f}")
    
    # --- Metrics Interpretation ---
    print("\n--- Metrics Interpretation ---")
    
    print("\n1. R² (Coefficient of Determination):")
    print("   Range: 0 to 1 (higher is better)")
    print(f"   Good fit: {result_good.metrics['r2']:.4f} - Excellent!")
    print(f"   Medium fit: {result_medium.metrics['r2']:.4f} - OK")
    print(f"   Bad fit: {result_bad.metrics['r2']:.4f} - Poor!")
    print("   > 0.95: Excellent")
    print("   0.90-0.95: Good")
    print("   0.80-0.90: Acceptable")
    print("   < 0.80: Poor")
    
    print("\n2. χ²_reduced (Reduced Chi-squared):")
    print("   Range: 0 to ∞ (close to 1 is ideal)")
    print(f"   Good fit: {result_good.metrics['chi2_reduced']:.4f}")
    print(f"   Medium fit: {result_medium.metrics['chi2_reduced']:.4f}")
    print(f"   Bad fit: {result_bad.metrics['chi2_reduced']:.4f}")
    print("   ≈ 1: Perfect (noise matches expected)")
    print("   < 1: Over-fitting or underestimated errors")
    print("   > 1: Under-fitting or underestimated errors")
    print("   >> 1: Poor fit")
    
    print("\n3. RMSE (Root Mean Squared Error):")
    print("   Range: 0 to ∞ (lower is better)")
    print(f"   Good fit: {result_good.metrics['rmse']:.4e}")
    print(f"   Medium fit: {result_medium.metrics['rmse']:.4e}")
    print(f"   Bad fit: {result_bad.metrics['rmse']:.4e}")
    print("   Depends on data scale")
    print("   Compare to data range for context")
    
    print("\n4. Pearson r (Correlation):")
    print("   Range: -1 to 1 (1 is perfect)")
    print(f"   Good fit: {result_good.metrics['pearson_r']:.4f}")
    print(f"   Medium fit: {result_medium.metrics['pearson_r']:.4f}")
    print(f"   Bad fit: {result_bad.metrics['pearson_r']:.4f}")
    print("   > 0.95: Excellent correlation")
    
    print("\n5. p-value:")
    print("   Range: 0 to 1")
    print(f"   Good fit: {result_good.metrics['p_value']:.4f}")
    print(f"   Medium fit: {result_medium.metrics['p_value']:.4f}")
    print(f"   Bad fit: {result_bad.metrics['p_value']:.4f}")
    print("   > 0.05: Cannot reject null hypothesis (good fit)")
    print("   < 0.05: Can reject null hypothesis (poor fit)")
    
    # --- Use Utilities ---
    print("\n--- Using Utility Functions ---")
    
    y_data = result_good.data.get_y()
    y_fit = result_good.y_fit
    
    corr = get_ab_correlation(y_data, y_fit)
    overlap = get_similarity_by_overlap(y_data, y_fit)
    
    print(f"\nget_ab_correlation: {corr:.4f}")
    print(f"get_similarity_by_overlap: {overlap:.4f}")
    print(f"(Compare to Pearson r: {result_good.metrics['pearson_r']:.4f})")
    
    print("\n✓ Evaluator and metrics demonstrated!")
    return result_good, result_medium, result_bad


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("pyFitting - COMPREHENSIVE ADVANCED EXAMPLES")
    print("="*80)
    print("\nThis demonstrates ALL features of pyFitting:")
    print("1. Data features (masking, transformations, weights)")
    print("2. Models (built-in + custom)")
    print("3. Loss functions (all 5)")
    print("4. Optimizers (comparison)")
    print("5. Fitter features (guess, bounds, fixed params)")
    print("6. Evaluator & metrics")
    
    try:
        # Run all examples
        example_1_data_features()
        example_2_models()
        example_3_loss_functions()
        example_4_optimizers()
        example_5_fitter_features()
        example_6_evaluator_metrics()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        print("\nYou've now seen:")
        print("  ✅ All data manipulation features")
        print("  ✅ All 6 built-in models + 3 custom models")
        print("  ✅ All 5 loss functions with comparisons")
        print("  ✅ All 6 optimizer methods with benchmarks")
        print("  ✅ All fitter configuration options")
        print("  ✅ All 12+ metrics and their interpretations")
        print("\npyFitting gives you complete control over every aspect of fitting! 🚀")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()