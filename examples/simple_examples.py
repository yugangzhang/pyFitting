"""
pyFitting - Simple Example

This example demonstrates basic usage of the pyFitting framework.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from pyFitting import (
    Fitter,
    ArrayData,
    GaussianModel,
    ExponentialModel,
    MSELoss,
    CorrelationLoss,
    LocalOptimizer
)


def example_1_gaussian():
    """Example 1: Fit a Gaussian peak"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Gaussian Peak Fitting")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 2.5 * np.exp(-0.5 * ((x - 5) / 1.2)**2) + 0.3
    y = y_true + np.random.normal(0, 0.05, len(x))
    
    # Create data object
    data = ArrayData(x, y)
    
    # Create model
    model = GaussianModel()
    
    # Fit (that's it!)
    fitter = Fitter(data, model)
    result = fitter.fit(verbose=True)
    
    
    # View results
    result.summary()
    
    print(f"\nTrue parameters: A=2.5, mu=5.0, sigma=1.2, c=0.3")
    print(f"Fitted parameters:")
    for name, value in result.parameters.values.items():
        print(f"  {name} = {value:.4f}")


    # Plot
    plot_fit(result, logx=True, logy=True, save='fit.png')
    plot_fit_with_residuals(result, save='analysis.png')
    plot_diagnostics(result, save='diagnostics.png')        
    
    return result


def example_2_exponential():
    """Example 2: Fit exponential decay"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Exponential Decay Fitting")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 5, 100)
    y_true = 3.0 * np.exp(-0.8 * x) + 0.2
    y = y_true + np.random.normal(0, 0.05, len(x))
    
    # Create data object
    data = ArrayData(x, y)
    
    # Create model with bounds
    model = ExponentialModel()
    model.set_bounds(
        A=(0, 10),
        k=(0, 5),
        c=(-1, 1)
    )
    
    # Fit with MSE loss
    loss = MSELoss(use_log=False)
    fitter = Fitter(data, model, loss=loss, optimizer='SLSQP')
    result = fitter.fit(verbose=True)
    
    # Results
    result.summary()
    
    print(f"\nTrue parameters: A=3.0, k=0.8, c=0.2")
    print(f"Fitted parameters:")
    for name, value in result.parameters.values.items():
        print(f"  {name} = {value:.4f}")
    
    return result


def example_3_custom_model():
    """Example 3: Custom model"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Model")
    print("="*70)
    
    # Import base model
    from pyFitting import BaseModel
    
    # Define custom model
    class DoubleExponentialModel(BaseModel):
        """
        Double exponential: y = A1*exp(-k1*x) + A2*exp(-k2*x) + c
        """
        def evaluate(self, x, A1, k1, A2, k2, c):
            return A1 * np.exp(-k1 * x) + A2 * np.exp(-k2 * x) + c
        
        def get_initial_guess(self, x, y):
            return {
                'A1': y.max() / 2,
                'k1': 0.5,
                'A2': y.max() / 2,
                'k2': 2.0,
                'c': y.min()
            }
    
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 5, 100)
    y_true = 2.0 * np.exp(-0.3 * x) + 1.0 * np.exp(-2.0 * x) + 0.1
    y = y_true + np.random.normal(0, 0.03, len(x))
    
    # Fit
    data = ArrayData(x, y)
    model = DoubleExponentialModel()
    model.set_bounds(
        A1=(0, 10), k1=(0, 5),
        A2=(0, 10), k2=(0, 5),
        c=(-1, 1)
    )
    
    fitter = Fitter(data, model)
    result = fitter.fit(verbose=True)
    
    result.summary()
    
    print(f"\nTrue parameters: A1=2.0, k1=0.3, A2=1.0, k2=2.0, c=0.1")
    
    return result


def example_4_compare_losses():
    """Example 4: Compare different loss functions"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Compare Loss Functions")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 2.5 * np.exp(-0.5 * ((x - 5) / 1.2)**2) + 0.3
    y = y_true + np.random.normal(0, 0.05, len(x))
    
    data = ArrayData(x, y)
    model = GaussianModel()
    
    # Try different losses
    losses = {
        'MSE (linear)': MSELoss(use_log=False),
        'MSE (log)': MSELoss(use_log=True),
        'Correlation': CorrelationLoss(use_log=True),
    }
    
    results = {}
    for name, loss in losses.items():
        print(f"\nFitting with {name}...")
        fitter = Fitter(data, model, loss=loss)
        result = fitter.fit(verbose=False)
        results[name] = result
        print(f"  R² = {result.metrics['r2']:.4f}")
        print(f"  R² (log) = {result.metrics['r2_log']:.4f}")
    
    print("\nAll loss functions give similar results for this problem!")
    
    return results


def example_5_compare_optimizers():
    """Example 5: Compare optimizers"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Compare Optimizers")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 5, 100)
    y_true = 3.0 * np.exp(-0.8 * x) + 0.2
    y = y_true + np.random.normal(0, 0.05, len(x))
    
    data = ArrayData(x, y)
    model = ExponentialModel()
    model.set_bounds(A=(0, 10), k=(0, 5), c=(-1, 1))
    
    # Try different optimizers
    methods = ['SLSQP', 'L-BFGS-B', 'Powell', 'trust-constr']
    
    print(f"\n{'Method':<15} {'Success':<10} {'R²':<10} {'nfev':<8}")
    print("-"*50)
    
    results = {}
    for method in methods:
        optimizer = LocalOptimizer(method)
        fitter = Fitter(data, model, optimizer=optimizer)
        result = fitter.fit(verbose=False)
        results[method] = result
        
        status = "✓" if result.success else "✗"
        print(f"{method:<15} {status:<10} {result.metrics['r2']:<10.4f} {result.nfev:<8}")
    
    print("\nSLSQP is recommended - good balance of speed and robustness!")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("pyFitting Examples")
    print("="*70)
    
    # Run all examples
    try:
        example_1_gaussian()
        example_2_exponential()
        example_3_custom_model()
        example_4_compare_losses()
        example_5_compare_optimizers()
        
        print("\n" + "="*70)
        print("All examples completed successfully! 🎉")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()